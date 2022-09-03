from typing import Iterable, List

import pytest

from jqlite.core.filters import (
    Array,
    Identity,
    Index,
    Literal,
    Pipe,
    Semi,
    Mul,
    Div,
    Add,
    Sub,
    Gt,
    Eq,
    Le,
    Lt,
    Ge,
    Ne,
    Object,
    Length,
    Map,
    String,
    Iteration,
    Slice,
    Range,
)
from jqlite.core.parser import (
    parse,
    Lexer,
    Token,
    TokenType,
    ParseError,
    OPERATORS,
    DOUBLE_OPERATORS,
)


def lex(text: str) -> List[Token]:
    return list(Lexer(text).lex())


def test_lex_empty():
    assert lex("") == []


def test_lex_ops():
    for op in OPERATORS:
        assert lex(op) == [Token(TokenType.OP, op)]
    with pytest.raises(ParseError, match="Unexpected character `@`."):
        lex("@")


def test_lex_double_ops():
    for op in DOUBLE_OPERATORS:
        assert lex(op) == [Token(TokenType.OP, op)]


def test_lex_number():
    assert lex("42") == [Token(TokenType.NUM, 42)]
    assert lex("42.0") == [Token(TokenType.NUM, 42.0)]
    assert lex("3.14") == [Token(TokenType.NUM, 3.14)]


def test_lex_ident():
    assert lex("foo") == [Token(TokenType.IDENT, "foo")]
    assert lex("_foo") == [Token(TokenType.IDENT, "_foo")]
    assert lex("foo123") == [Token(TokenType.IDENT, "foo123")]


def test_lexing_plain_string():
    assert lex('""') == [Token(TokenType.STR, "")]
    assert lex('"abc"') == [Token(TokenType.STR, "abc")]


def test_lexing_unclosed_string():
    with pytest.raises(ParseError, match="Unclosed string."):
        lex('"abc')


def test_lexing_string_interp():
    assert lex('"a{}c"') == [
        Token(TokenType.STR_START, "a"),
        Token(TokenType.OP, "{"),
        Token(TokenType.OP, "}"),
        Token(TokenType.STR_END, "c"),
    ]

    assert lex('"a{}b{}c"') == [
        Token(TokenType.STR_START, "a"),
        Token(TokenType.OP, "{"),
        Token(TokenType.OP, "}"),
        Token(TokenType.STR, "b"),
        Token(TokenType.OP, "{"),
        Token(TokenType.OP, "}"),
        Token(TokenType.STR_END, "c"),
    ]

    assert lex('"{}"') == [
        Token(TokenType.STR_START, ""),
        Token(TokenType.OP, "{"),
        Token(TokenType.OP, "}"),
        Token(TokenType.STR_END, ""),
    ]


def test_lexing_string_interp_nested():
    assert lex('"{ "abc" }"') == [
        Token(TokenType.STR_START, ""),
        Token(TokenType.OP, "{"),
        Token(TokenType.STR, "abc"),
        Token(TokenType.OP, "}"),
        Token(TokenType.STR_END, ""),
    ]

    assert lex('"{ "a{ "b" }c" }"') == [
        Token(TokenType.STR_START, ""),
        Token(TokenType.OP, "{"),
        Token(TokenType.STR_START, "a"),
        Token(TokenType.OP, "{"),
        Token(TokenType.STR, "b"),
        Token(TokenType.OP, "}"),
        Token(TokenType.STR_END, "c"),
        Token(TokenType.OP, "}"),
        Token(TokenType.STR_END, ""),
    ]

    assert lex('"{ {"a": 1} }"') == [
        Token(TokenType.STR_START, ""),
        Token(TokenType.OP, "{"),
        Token(TokenType.OP, "{"),
        Token(TokenType.STR, "a"),
        Token(TokenType.OP, ":"),
        Token(TokenType.NUM, 1.0),
        Token(TokenType.OP, "}"),
        Token(TokenType.OP, "}"),
        Token(TokenType.STR_END, ""),
    ]


def test_lexing_unterminated_string_interp():
    with pytest.raises(
        ParseError, match="Unterminated expression in string interpolation."
    ):
        lex('"{')


def test_lex_token_separation():
    assert lex("123abc") == [
        Token(TokenType.NUM, 123),
        Token(TokenType.IDENT, "abc"),
    ]
    assert lex("123 abc") == [
        Token(TokenType.NUM, 123),
        Token(TokenType.IDENT, "abc"),
    ]


def test_lex_whitespace():
    assert lex(" \r\n\t") == []
    assert lex(" 42  ") == [Token(TokenType.NUM, 42)]


def test_parse_empty_expr():
    assert parse("") is None
    assert parse(" \r\n\t") is None


def test_parse_identity():
    assert parse(".") == Identity()


def test_parse_iterate():
    assert parse(".[]") == Iteration()


def test_parse_prop():
    assert parse(".[1]") == Index(Literal(1))
    assert parse(".foo") == Index(Literal("foo"))
    assert parse("._foo") == Index(Literal("_foo"))
    assert parse("._foo_123") == Index(Literal("_foo_123"))


def test_parse_slice():
    assert parse(".[:]") == Slice([Literal(None), Literal(None)])
    assert parse(".[1:]") == Slice([Literal(1), Literal(None)])
    assert parse(".[:2]") == Slice([Literal(None), Literal(2)])
    assert parse(".[1:2]") == Slice([Literal(1), Literal(2)])
    assert parse(".[::]") == Slice([Literal(None), Literal(None), Literal(None)])
    assert parse(".[1::]") == Slice([Literal(1), Literal(None), Literal(None)])
    assert parse(".[:1:]") == Slice([Literal(None), Literal(1), Literal(None)])
    assert parse(".[::1]") == Slice([Literal(None), Literal(None), Literal(1)])
    assert parse(".[1:1:]") == Slice([Literal(1), Literal(1), Literal(None)])
    assert parse(".[:1:1]") == Slice([Literal(None), Literal(1), Literal(1)])
    assert parse(".[1::1]") == Slice([Literal(1), Literal(None), Literal(1)])
    assert parse(".[1:2:1]") == Slice([Literal(1), Literal(2), Literal(1)])

    with pytest.raises(ValueError):
        parse(".[:::]")


def test_parse_literal():
    assert parse("null") == Literal(None)
    assert parse("true") == Literal(True)
    assert parse("false") == Literal(False)
    assert parse("3.14") == Literal(3.14)


def test_parse_array():
    assert parse("[]") == Array([])
    assert parse("[.]") == Array([Identity()])
    assert parse("[1, 2, 3]") == Array([Literal(1), Literal(2), Literal(3)])
    assert parse("[.[] | [.]]") == Array([Pipe([Iteration(), Array([Identity()])])])


def test_parse_object():
    assert parse("{}") == Object([])
    assert parse('{"foo": 42}') == Object([(Literal("foo"), Literal(42))])
    assert parse("{foo}") == Object([(Literal("foo"), Index(Literal("foo")))])
    assert parse('{"foo"}') == Object([(Literal("foo"), Index(Literal("foo")))])
    assert parse('{["foo" + "bar"]: 42}') == Object(
        [(Add(String([Literal("foo")]), String([Literal("bar")])), Literal(42))]
    )
    assert parse('{"foo": [.[]]}') == Object([(Literal("foo"), Array([Iteration()]))])
    assert parse('{"foo": 1, "bar": 2}') == Object(
        [(Literal("foo"), Literal(1)), (Literal("bar"), Literal(2))]
    )
    assert parse('{"foo": 1, "bar": 2,}') == Object(
        [(Literal("foo"), Literal(1)), (Literal("bar"), Literal(2))]
    )


def test_parse_string():
    assert parse('"foo"') == String([Literal("foo")])


def test_parse_string_interpolation():
    assert parse('"a{1 + 1}c"') == String(
        [Literal("a"), Add(Literal(1), Literal(1)), Literal("c")]
    )

    assert parse('"{ "a" }{ "b" }"') == String(
        [String([Literal("a")]), String([Literal("b")])]
    )

    assert parse('"{ "a{1 + 1}c" }"') == String(
        [String([Literal("a"), Add(Literal(1), Literal(1)), Literal("c")])]
    )


def test_parse_mul():
    assert parse("1 * 2") == Mul(Literal(1), Literal(2))
    assert parse(".[] / 2") == Div(Iteration(), Literal(2))
    assert parse("1 * 2 * 3") == Mul(Mul(Literal(1), Literal(2)), Literal(3))


def test_parse_add():
    assert parse("1 + 2") == Add(Literal(1), Literal(2))
    assert parse("1 - 2") == Sub(Literal(1), Literal(2))
    assert parse("1 + 2 + 3") == Add(Add(Literal(1), Literal(2)), Literal(3))
    # fmt: off
    assert parse("1 + 2 * 3 - 4") == Sub(
        Add(
            Literal(1),
            Mul(
                Literal(2),
                Literal(3)
            )
        ),
        Literal(4)
    )
    # fmt: on


def test_parse_eq():
    assert parse("1 > 2") == Gt(Literal(1), Literal(2))
    assert parse("1 >= 2") == Ge(Literal(1), Literal(2))
    assert parse("1 < 2") == Lt(Literal(1), Literal(2))
    assert parse("1 <= 2") == Le(Literal(1), Literal(2))
    assert parse("1 == 2") == Eq(Literal(1), Literal(2))
    assert parse("1 != 2") == Ne(Literal(1), Literal(2))
    # fmt: off
    assert parse("1 + 2 >= 3 - 4") == Ge(
        Add(Literal(1), Literal(2)),
        Sub(Literal(3), Literal(4))
    )
    # fmt: on


def test_parse_semi():
    assert parse(".; .foo;null") == Semi(
        [Identity(), Index(Literal("foo")), Literal(None)]
    )


def test_parse_pipe():
    assert parse(".foo | .bar") == Pipe([Index(Literal("foo")), Index(Literal("foo"))])
    assert parse(".foo | .bar | .baz") == Pipe(
        [Index(Literal("foo")), Index(Literal("foo")), Index(Literal("foo"))]
    )


def test_parse_fn_call():
    assert parse("length") == Length()
    assert parse("range(1, 2)") == Range(Literal(1), Literal(2))
    assert parse("map(. * 2)") == Map(Mul(Identity(), Literal(2)))
