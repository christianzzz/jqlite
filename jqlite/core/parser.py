from enum import Enum
from typing import Any, Iterable, NamedTuple, Optional

from jqlite.core.context import Context
from jqlite.core.filters import (
    Filter,
    Literal,
    Index,
    Identity,
    Array,
    Semi,
    Mul,
    Div,
    Add,
    Sub,
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
    Ne,
    Pipe,
    Object,
    Fn,
    Mod,
    String,
    And,
    Or,
    Neg,
    Pos,
    Not,
    Iteration,
    Slice,
)


class TokenType(Enum):
    PUNCT = "punct"
    NUM = "num"
    STR = "str"
    STR_START = "str_start"
    STR_END = "str_end"
    IDENT = "ident"


class Token(NamedTuple):
    type: TokenType
    val: Any


class Lexer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.mode_stack = []

    def lex(self) -> Iterable[Token]:
        while self.pos < len(self.text):
            if self.mode_stack and self.text[self.mode_stack[-1]] == '"':
                token_type = TokenType.STR

                if self.pos == self.mode_stack[-1]:
                    token_type = TokenType.STR_START
                    self.pos += 1

                start = self.pos
                while (
                    self.pos < len(self.text)
                    and self.text[self.pos] != '"'
                    and self.text[self.pos] != "{"
                ):
                    self.pos += 1

                if self.pos < len(self.text) and self.text[self.pos] == "{":
                    yield Token(token_type, self.text[start : self.pos])
                    yield Token(TokenType.PUNCT, "{")
                    self.mode_stack.append(self.pos)
                    self.pos += 1
                elif self.pos < len(self.text) and self.text[self.pos] == '"':
                    if token_type == TokenType.STR_START:
                        token_type = TokenType.STR
                    else:
                        token_type = TokenType.STR_END
                    yield Token(token_type, self.text[start : self.pos])
                    self.mode_stack.pop()
                    self.pos += 1
                else:
                    raise Exception("Unexpected end of string")
            else:
                char = self.text[self.pos]

                if char.isspace():
                    self.pos += 1
                    continue
                elif char in "!<>=+-*/" and self.text[self.pos + 1] == "=":
                    self.pos += 2
                    yield Token(TokenType.PUNCT, char + "=")
                elif char == "{":
                    yield Token(TokenType.PUNCT, char)
                    self.mode_stack.append(self.pos)
                    self.pos += 1
                elif char == "}":
                    yield Token(TokenType.PUNCT, char)
                    self.mode_stack.pop()
                    self.pos += 1
                elif char in ".,:;[]()<>=+-*/%|":
                    self.pos += 1
                    yield Token(TokenType.PUNCT, char)
                elif char == '"':
                    self.mode_stack.append(self.pos)
                elif char.isdigit():
                    yield self._read_num()
                elif char.isalpha() or char == "_":
                    yield self._read_ident()
                else:
                    raise ValueError(f"invalid character {char}")

    def _read_num(self):
        start = self.pos
        while self.pos < len(self.text) and (
            self.text[self.pos] == "." or self.text[self.pos].isdigit()
        ):
            self.pos += 1
        return Token(TokenType.NUM, float(self.text[start : self.pos]))

    def _read_ident(self):
        start = self.pos

        while self.pos < len(self.text):
            char = self.text[self.pos]
            if not char.isalnum() and char != "_":
                break
            self.pos += 1

        return Token(TokenType.IDENT, self.text[start : self.pos])


class Parser:
    """
    优先级:
        |
        ,
        = +=, -=, *=, /=
        > >= < <= == !=
        + -
        * /
        () atom
    """

    def __init__(self, lexer: Lexer):
        self.ctx = Context()
        self.tokens = list(lexer.lex())
        self.pos = 0

    def parse(self) -> Optional[Filter]:
        if not self._peek():
            return
        return self._parse_pipe()

    def _parse_pipe(self) -> Filter:
        filters = [self._parse_semi()]
        while self._peek() == Token(TokenType.PUNCT, "|"):
            self._next()
            filters.append(self._parse_semi())
        return Pipe(filters) if len(filters) > 1 else filters[0]

    def _parse_semi(self) -> Filter:
        filters = [self._parse_logical_or()]
        while self._peek() == Token(TokenType.PUNCT, ";"):
            self._next()
            filters.append(self._parse_logical_or())
        return Semi(filters) if len(filters) > 1 else filters[0]

    def _parse_logical_or(self):
        left = self._parse_logical_and()
        while self._peek() == Token(TokenType.IDENT, "or"):
            self._next()
            right = self._parse_logical_and()
            left = Or(left, right)
        return left

    def _parse_logical_and(self):
        left = self._parse_eq()
        while self._peek() == Token(TokenType.IDENT, "and"):
            self._next()
            right = self._parse_eq()
            left = And(left, right)
        return left

    def _parse_eq(self):
        result = self._parse_add()
        if self._peek() == Token(TokenType.PUNCT, ">"):
            self._next()
            result = Gt(result, self._parse_add())
        elif self._peek() == Token(TokenType.PUNCT, ">="):
            self._next()
            result = Ge(result, self._parse_add())
        elif self._peek() == Token(TokenType.PUNCT, "<"):
            self._next()
            result = Lt(result, self._parse_add())
        elif self._peek() == Token(TokenType.PUNCT, "<="):
            self._next()
            result = Le(result, self._parse_add())
        elif self._peek() == Token(TokenType.PUNCT, "=="):
            self._next()
            result = Eq(result, self._parse_add())
        elif self._peek() == Token(TokenType.PUNCT, "!="):
            self._next()
            result = Ne(result, self._parse_add())
        return result

    def _parse_add(self) -> Filter:
        result = self._parse_mul()
        while True:
            if self._peek() == Token(TokenType.PUNCT, "+"):
                self._next()
                result = Add(result, self._parse_mul())
            elif self._peek() == Token(TokenType.PUNCT, "-"):
                self._next()
                result = Sub(result, self._parse_mul())
            else:
                break
        return result

    def _parse_mul(self) -> Filter:
        result = self._parse_unary()
        while True:
            if self._peek() == Token(TokenType.PUNCT, "*"):
                self._next()
                result = Mul(result, self._parse_unary())
            elif self._peek() == Token(TokenType.PUNCT, "/"):
                self._next()
                result = Div(result, self._parse_unary())
            elif self._peek() == Token(TokenType.PUNCT, "%"):
                self._next()
                result = Mod(result, self._parse_unary())
            else:
                break
        return result

    def _parse_unary(self) -> Filter:
        token = self._peek()
        if token == Token(TokenType.PUNCT, "-"):
            self._next()
            return Neg(self._parse_unary())
        elif token == Token(TokenType.PUNCT, "+"):
            self._next()
            return Pos(self._parse_unary())
        elif token == Token(TokenType.IDENT, "not"):
            self._next()
            return Not(self._parse_unary())
        else:
            return self._parse_primary()

    def _parse_primary(self) -> Filter:
        token = self._peek()
        result = None
        if token == Token(TokenType.PUNCT, "("):
            self._next()
            result = self._parse_pipe()
            self._expect(Token(TokenType.PUNCT, ")"))
        elif (
            token.val == "."
            and self._peek(1)
            and self._peek(1) == Token(TokenType.PUNCT, "[")
        ):
            self.pos += 1
        elif (
            token.val == "." and self._peek(1) and self._peek(1).type == TokenType.IDENT
        ):
            pass
        elif token.val == ".":
            self._next()
            result = Identity()
        elif token == Token(TokenType.PUNCT, "["):
            result = self._parse_array()
        elif token == Token(TokenType.PUNCT, "{"):
            result = self._parse_object()
        elif token.type == TokenType.IDENT:
            if token.val == "null":
                self._next()
                result = Literal(None)
            elif token.val == "true":
                self._next()
                result = Literal(True)
            elif token.val == "false":
                self._next()
                result = Literal(False)
            else:
                result = self._parse_fn_call()
        elif token.type == TokenType.NUM:
            self._next()
            result = Literal(token.val)
        elif token.type == TokenType.STR:
            self._next()
            result = String([Literal(token.val)])
        elif token.type == TokenType.STR_START:
            result = self._parse_string_interpolation()
        else:
            raise ValueError(f"invalid token {self.tokens[self.pos]}")

        while True:
            if self._peek() == Token(TokenType.PUNCT, "["):
                indices = []

                self._next()
                if self._peek() != Token(TokenType.PUNCT, "]"):
                    indices.append(self._parse_slice_index())
                    for _ in range(2):
                        if self._peek() == Token(TokenType.PUNCT, ":"):
                            self._next()
                            indices.append(self._parse_slice_index())
                self._expect(Token(TokenType.PUNCT, "]"))

                if not indices:
                    index = Iteration()
                elif len(indices) == 1:
                    index = Index(indices[0])
                else:
                    index = Slice(indices)

                result = Pipe([result, index]) if result else index
            elif self._peek() == Token(TokenType.PUNCT, "."):
                self._next()
                if self._peek() and self._peek().type == TokenType.IDENT:
                    result = (
                        Pipe([result, Index(Literal(self._next().val))])
                        if result
                        else Index(Literal(self._next().val))
                    )
            else:
                break

        return result

    def _parse_slice_index(self) -> Filter:
        if self._peek() == Token(TokenType.PUNCT, ":") or self._peek() == Token(
            TokenType.PUNCT, "]"
        ):
            return Literal(None)
        else:
            return self._parse_pipe()

    def _parse_fn_call(self) -> Fn:
        name = self._peek().val
        fn = self.ctx.get(name)
        if not fn:
            raise ValueError(f"{name} undefined")

        self._next()
        args = []
        if self._peek() == Token(TokenType.PUNCT, "("):
            self._next()
            if self._peek() == Token(TokenType.PUNCT, ")"):
                self._next()
            else:
                args.append(self._parse_pipe())
                while self._peek() == Token(TokenType.PUNCT, ","):
                    self._next()
                    args.append(self._parse_pipe())
                self._next()
        else:
            pass
        return fn(*args)

    def _parse_string_interpolation(self) -> Filter:
        filters = []
        token = self._next()
        if token.val:
            filters.append(Literal(token.val))
        while self.pos < len(self.tokens):
            token = self._peek()
            if token == Token(TokenType.PUNCT, "{"):
                self._next()
                filters.append(self._parse_pipe())
                self._expect(Token(TokenType.PUNCT, "}"))
            elif token.type == TokenType.STR_START:
                self._parse_string_interpolation()
            elif token.type == TokenType.STR:
                self._next()
                if token.val:
                    filters.append(Literal(token.val))
            elif token.type == TokenType.STR_END:
                self._next()
                if token.val:
                    filters.append(Literal(token.val))
                break
            else:
                raise ValueError(f"invalid token {token}")
        return String(filters)

    def _parse_array(self) -> Filter:
        result = []
        self._expect(Token(TokenType.PUNCT, "["))
        while True:
            if self._peek() == Token(TokenType.PUNCT, "]"):
                self._next()
                break
            else:
                result.append(self._parse_pipe())
                if self._peek() == Token(TokenType.PUNCT, ","):
                    self._next()
        return Array(result)

    def _parse_object(self) -> Filter:
        self._expect(Token(TokenType.PUNCT, "{"))
        if self._peek() == Token(TokenType.PUNCT, "}"):
            self._next()
            return Object([])

        result = []
        while True:
            result.append(self._parse_object_item())

            if self._peek() != Token(TokenType.PUNCT, ",") and self._peek() != Token(
                TokenType.PUNCT, "}"
            ):
                raise ValueError(f"Unexpected token {self._peek()}")

            if self._peek() == Token(TokenType.PUNCT, ","):
                self._next()

            if self._peek() == Token(TokenType.PUNCT, "}"):
                self._next()
                break

        return Object(result)

    def _parse_object_item(self):
        if self._peek() == Token(TokenType.PUNCT, "["):
            self._next()
            key = self._parse_pipe()
            self._expect(Token(TokenType.PUNCT, "]"))
            if self._peek() == Token(TokenType.PUNCT, ":"):
                self._next()
                return key, self._parse_pipe()
            else:
                raise ValueError(f"Unexpected token {self._peek()}")
        elif self._peek().type == TokenType.STR or self._peek().type == TokenType.IDENT:
            key = self._peek().val
            self._next()
            if self._peek() == Token(TokenType.PUNCT, ":"):
                self._next()
                return Literal(key), self._parse_pipe()
            elif self._peek() == Token(TokenType.PUNCT, ",") or self._peek() == Token(
                TokenType.PUNCT, "}"
            ):
                return Literal(key), Index(Literal(key))
            else:
                raise ValueError(f"Unexpected token {self._peek()}")
        else:
            raise ValueError(f"Unexpected token {self._peek()}")

    def _peek(self, n: int = 0) -> Optional[Token]:
        if self.pos + n >= len(self.tokens):
            return
        return self.tokens[self.pos + n]

    def _next(self) -> Optional[Token]:
        token = self._peek()
        if token:
            self.pos += 1
        return token

    def _expect(self, token: Token):
        t = self._next()
        if t != token:
            raise ValueError(f"Expect {token}, got {t}")


def parse(expr: str) -> Optional[Filter]:
    lexer = Lexer(expr)
    parser = Parser(lexer)
    return parser.parse()
