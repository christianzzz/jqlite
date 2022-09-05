from enum import Enum, auto
from typing import Any, Iterable, NamedTuple, Optional, Union

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
    OP = auto()
    NUM = auto()
    STR = auto()
    STR_START = auto()
    STR_END = auto()
    IDENT = auto()


class Token(NamedTuple):
    type: TokenType
    value: Union[str, int, float]


class ParseError(Exception):
    pass


OPERATORS = set(".,:;[]()<>=+-*/%|")
DOUBLE_OPERATORS = {">=", "<=", "==", "!=", "+=", "-=", "*=", "/="}


class Lexer:
    def __init__(self, text: str):
        self.text = text
        self.index = 0
        self.mode_stack = []

    def lex(self) -> Iterable[Token]:
        return self._read_expr()

    def _read_expr(self, nested: bool = False) -> Iterable[Token]:
        while self.index < len(self.text):
            char = self.text[self.index]
            if char.isspace():
                self.index += 1
                continue
            elif char == "{":
                yield Token(TokenType.OP, char)
                self.index += 1
                if nested:
                    self.mode_stack.append("{")
            elif char == "}":
                yield Token(TokenType.OP, char)
                self.index += 1
                if nested:
                    self.mode_stack.pop()
                    if self.mode_stack and self.mode_stack[-1] == '"':
                        break
            elif self.text[self.index : self.index + 2] in DOUBLE_OPERATORS:
                self.index += 2
                yield Token(TokenType.OP, char + "=")
            elif char in OPERATORS:
                self.index += 1
                yield Token(TokenType.OP, char)
            elif char == '"':
                yield from self._read_string()
            elif char.isdigit():
                yield self._read_num()
            elif self._is_ident(char):
                yield self._read_ident()
            else:
                raise ParseError(f"Unexpected character `{char}`.")

        if nested and self.mode_stack and self.mode_stack[-1] == "{":
            raise ParseError("Unterminated expression in string interpolation.")

    def _read_string(self) -> Iterable[Token]:
        self.mode_stack.append('"')
        self.index += 1

        token_type = TokenType.STR_START
        start = self.index
        while self.index < len(self.text):
            char = self.text[self.index]
            if char == "{":
                yield Token(token_type, self.text[start : self.index])
                yield from self._read_expr(nested=True)
                start = self.index
                token_type = TokenType.STR
            elif char == '"':
                yield Token(
                    TokenType.STR
                    if token_type == TokenType.STR_START
                    else TokenType.STR_END,
                    self.text[start : self.index],
                )
                self.mode_stack.pop()
                self.index += 1
                break
            else:
                self.index += 1

        if self.mode_stack and self.mode_stack[-1] == '"':
            raise ParseError("Unclosed string.")

    def _read_num(self):
        start = self.index
        while self.index < len(self.text) and (
            self.text[self.index] == "." or self.text[self.index].isdigit()
        ):
            self.index += 1
        return Token(TokenType.NUM, float(self.text[start : self.index]))

    @staticmethod
    def _is_ident(char: str) -> bool:
        return char.isalpha() or char == "_"

    def _read_ident(self):
        start = self.index

        while self.index < len(self.text):
            char = self.text[self.index]
            if not char.isalnum() and char != "_":
                break
            self.index += 1

        return Token(TokenType.IDENT, self.text[start : self.index])


class Parser:
    def __init__(self, lexer: Lexer):
        self.ctx = Context()
        self.tokens = list(lexer.lex())
        self.index = 0

    def parse(self) -> Optional[Filter]:
        if not self._peek():
            return
        return self._parse_pipe()

    def _parse_pipe(self) -> Filter:
        left = self._parse_semi()
        while self._peek() == Token(TokenType.OP, "|"):
            self._next()
            left = Pipe(left, self._parse_semi())
        return left

    def _parse_semi(self) -> Filter:
        filters = [self._parse_logical_or()]
        while self._peek() == Token(TokenType.OP, ";"):
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
        if self._peek() == Token(TokenType.OP, ">"):
            self._next()
            result = Gt(result, self._parse_add())
        elif self._peek() == Token(TokenType.OP, ">="):
            self._next()
            result = Ge(result, self._parse_add())
        elif self._peek() == Token(TokenType.OP, "<"):
            self._next()
            result = Lt(result, self._parse_add())
        elif self._peek() == Token(TokenType.OP, "<="):
            self._next()
            result = Le(result, self._parse_add())
        elif self._peek() == Token(TokenType.OP, "=="):
            self._next()
            result = Eq(result, self._parse_add())
        elif self._peek() == Token(TokenType.OP, "!="):
            self._next()
            result = Ne(result, self._parse_add())
        return result

    def _parse_add(self) -> Filter:
        result = self._parse_mul()
        while True:
            if self._peek() == Token(TokenType.OP, "+"):
                self._next()
                result = Add(result, self._parse_mul())
            elif self._peek() == Token(TokenType.OP, "-"):
                self._next()
                result = Sub(result, self._parse_mul())
            else:
                break
        return result

    def _parse_mul(self) -> Filter:
        result = self._parse_unary()
        while True:
            if self._peek() == Token(TokenType.OP, "*"):
                self._next()
                result = Mul(result, self._parse_unary())
            elif self._peek() == Token(TokenType.OP, "/"):
                self._next()
                result = Div(result, self._parse_unary())
            elif self._peek() == Token(TokenType.OP, "%"):
                self._next()
                result = Mod(result, self._parse_unary())
            else:
                break
        return result

    def _parse_unary(self) -> Filter:
        token = self._peek()
        if token == Token(TokenType.OP, "-"):
            self._next()
            return Neg(self._parse_unary())
        elif token == Token(TokenType.OP, "+"):
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
        if token == Token(TokenType.OP, "("):
            self._next()
            result = self._parse_pipe()
            self._expect(Token(TokenType.OP, ")"))
        elif (
            token.value == "."
            and self._peek(1)
            and self._peek(1) == Token(TokenType.OP, "[")
        ):
            self.index += 1
        elif (
            token.value == "."
            and self._peek(1)
            and (
                self._is_string(self._peek(1)) or self._peek(1).type == TokenType.IDENT
            )
        ):
            pass
        elif token.value == ".":
            self._next()
            result = Identity()
        elif token == Token(TokenType.OP, "["):
            result = self._parse_array()
        elif token == Token(TokenType.OP, "{"):
            result = self._parse_object()
        elif token.type == TokenType.IDENT:
            if token.value == "null":
                self._next()
                result = Literal(None)
            elif token.value == "true":
                self._next()
                result = Literal(True)
            elif token.value == "false":
                self._next()
                result = Literal(False)
            else:
                result = self._parse_fn_call()
        elif token.type == TokenType.NUM:
            self._next()
            result = Literal(token.value)
        elif self._is_string(token):
            result = self._parse_string()
        else:
            raise ParseError(f"Unexpected token: {self.tokens[self.index]}")

        while True:
            if self._peek() == Token(TokenType.OP, "["):
                indices = []

                self._next()
                if self._peek() != Token(TokenType.OP, "]"):
                    indices.append(self._parse_slice_index())
                    for _ in range(2):
                        if self._peek() == Token(TokenType.OP, ":"):
                            self._next()
                            indices.append(self._parse_slice_index())
                self._expect(Token(TokenType.OP, "]"))

                if not indices:
                    index = Iteration()
                elif len(indices) == 1:
                    index = Index(indices[0])
                else:
                    index = Slice(indices)

                result = Pipe([result, index]) if result else index
            elif self._peek() == Token(TokenType.OP, "."):
                self._next()
                if self._peek() and self._peek().type == TokenType.IDENT:
                    result = (
                        Pipe([result, Index(Literal(self._next().value))])
                        if result
                        else Index(Literal(self._next().value))
                    )
                elif self._peek() and self._is_string(self._peek()):
                    result = (
                        Pipe([result, Index(self._parse_string())])
                        if result
                        else Index(self._parse_string())
                    )
            else:
                break

        return result

    def _parse_slice_index(self) -> Filter:
        if self._peek() == Token(TokenType.OP, ":") or self._peek() == Token(
            TokenType.OP, "]"
        ):
            return Literal(None)
        else:
            return self._parse_pipe()

    def _parse_fn_call(self) -> Fn:
        name = self._peek().value
        fn = self.ctx.get(name)
        if not fn:
            raise ValueError(f"{name} undefined")

        self._next()
        args = []
        if self._peek() == Token(TokenType.OP, "("):
            self._next()
            if self._peek() == Token(TokenType.OP, ")"):
                self._next()
            else:
                args.append(self._parse_pipe())
                while self._peek() == Token(TokenType.OP, ","):
                    self._next()
                    args.append(self._parse_pipe())
                self._next()
        else:
            pass
        return fn(*args)

    def _parse_string(self) -> Filter:
        token = self._peek()
        if token.type == TokenType.STR:
            self._next()
            return String([Literal(token.value)])
        elif token.type == TokenType.STR_START:
            return self._parse_string_interp()

    def _parse_string_interp(self) -> Filter:
        filters = []
        token = self._next()
        if token.value:
            filters.append(Literal(token.value))
        while self.index < len(self.tokens):
            token = self._peek()
            if token == Token(TokenType.OP, "{"):
                self._next()
                filters.append(self._parse_pipe())
                self._expect(Token(TokenType.OP, "}"))
            elif token.type == TokenType.STR_START:
                self._parse_string()
            elif token.type == TokenType.STR:
                self._next()
                if token.value:
                    filters.append(Literal(token.value))
            elif token.type == TokenType.STR_END:
                self._next()
                if token.value:
                    filters.append(Literal(token.value))
                break
            else:
                raise ValueError(f"invalid token {token}")
        return String(filters)

    def _parse_array(self) -> Filter:
        result = []
        self._expect(Token(TokenType.OP, "["))
        while True:
            if self._peek() == Token(TokenType.OP, "]"):
                self._next()
                break
            else:
                result.append(self._parse_pipe())
                if self._peek() == Token(TokenType.OP, ","):
                    self._next()
        return Array(result)

    def _parse_object(self) -> Filter:
        self._expect(Token(TokenType.OP, "{"))
        if self._peek() == Token(TokenType.OP, "}"):
            self._next()
            return Object([])

        result = []
        while True:
            result.append(self._parse_object_item())

            if self._peek() != Token(TokenType.OP, ",") and self._peek() != Token(
                TokenType.OP, "}"
            ):
                raise ParseError(f"Unexpected token {self._peek()}")

            if self._peek() == Token(TokenType.OP, ","):
                self._next()

            if self._peek() == Token(TokenType.OP, "}"):
                self._next()
                break

        return Object(result)

    def _parse_object_item(self):
        if self._peek() == Token(TokenType.OP, "["):
            self._next()
            key = self._parse_pipe()
            self._expect(Token(TokenType.OP, "]"))
            if self._peek() == Token(TokenType.OP, ":"):
                self._next()
                return key, self._parse_pipe()
            else:
                raise ParseError(f"Unexpected token {self._peek()}")
        elif self._is_string(self._peek()) or self._peek().type == TokenType.IDENT:
            if self._is_string(self._peek()):
                key = self._parse_string()
            else:
                key = String([Literal(self._next().value)])
            if self._peek() == Token(TokenType.OP, ":"):
                self._next()
                return key, self._parse_pipe()
            elif self._peek() == Token(TokenType.OP, ",") or self._peek() == Token(
                TokenType.OP, "}"
            ):
                return key, Index(key)
            else:
                raise ParseError(f"Unexpected token {self._peek()}")
        else:
            raise ParseError(f"Unexpected token {self._peek()}")

    @staticmethod
    def _is_string(token: Token) -> bool:
        return token.type == TokenType.STR or token.type == TokenType.STR_START

    def _peek(self, n: int = 0) -> Optional[Token]:
        if self.index + n >= len(self.tokens):
            return
        return self.tokens[self.index + n]

    def _next(self) -> Optional[Token]:
        token = self._peek()
        if token:
            self.index += 1
        return token

    def _expect(self, token: Token):
        t = self._next()
        if t != token:
            raise ParseError(f"Expect {token}, got {t}")


def parse(expr: str) -> Optional[Filter]:
    lexer = Lexer(expr)
    parser = Parser(lexer)
    return parser.parse()
