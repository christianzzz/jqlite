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
    Iterator,
    Slice,
    Range,
)
from jqlite.core.parser import parse, Lexer, Token, TokenType


def test_lex_string_interpolation():
    lexer = Lexer('"abc"')
    assert list(lexer.lex()) == [Token(TokenType.STR, "abc")]

    lexer = Lexer('"a{}c"')
    assert list(lexer.lex()) == [
        Token(TokenType.STR_START, "a"),
        Token(TokenType.PUNCT, "{"),
        Token(TokenType.PUNCT, "}"),
        Token(TokenType.STR_END, "c"),
    ]

    lexer = Lexer('"a{}b{}c"')
    assert list(lexer.lex()) == [
        Token(TokenType.STR_START, "a"),
        Token(TokenType.PUNCT, "{"),
        Token(TokenType.PUNCT, "}"),
        Token(TokenType.STR, "b"),
        Token(TokenType.PUNCT, "{"),
        Token(TokenType.PUNCT, "}"),
        Token(TokenType.STR_END, "c"),
    ]

    lexer = Lexer('"{}"')
    assert list(lexer.lex()) == [
        Token(TokenType.STR_START, ""),
        Token(TokenType.PUNCT, "{"),
        Token(TokenType.PUNCT, "}"),
        Token(TokenType.STR_END, ""),
    ]

    lexer = Lexer('"{ "abc" }"')
    assert list(lexer.lex()) == [
        Token(TokenType.STR_START, ""),
        Token(TokenType.PUNCT, "{"),
        Token(TokenType.STR, "abc"),
        Token(TokenType.PUNCT, "}"),
        Token(TokenType.STR_END, ""),
    ]

    lexer = Lexer('"{ "a{"b"}c" }"')
    assert list(lexer.lex()) == [
        Token(TokenType.STR_START, ""),
        Token(TokenType.PUNCT, "{"),
        Token(TokenType.STR_START, "a"),
        Token(TokenType.PUNCT, "{"),
        Token(TokenType.STR, "b"),
        Token(TokenType.PUNCT, "}"),
        Token(TokenType.STR_END, "c"),
        Token(TokenType.PUNCT, "}"),
        Token(TokenType.STR_END, ""),
    ]

    lexer = Lexer('"{ {"a": 1} }"')
    assert list(lexer.lex()) == [
        Token(TokenType.STR_START, ""),
        Token(TokenType.PUNCT, "{"),
        Token(TokenType.PUNCT, "{"),
        Token(TokenType.STR, "a"),
        Token(TokenType.PUNCT, ":"),
        Token(TokenType.NUM, 1.0),
        Token(TokenType.PUNCT, "}"),
        Token(TokenType.PUNCT, "}"),
        Token(TokenType.STR_END, ""),
    ]


def test_parse_empty_expr():
    assert parse("") is None
    assert parse(" \r\n\t") is None


def test_parse_identity():
    assert parse(".") == Identity()


def test_parse_iterate():
    assert parse(".[]") == Iterator()


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
    assert parse("[.[] | [.]]") == Array([Pipe([Iterator(), Array([Identity()])])])


def test_parse_object():
    assert parse("{}") == Object([])
    assert parse('{"foo": 42}') == Object([(Literal("foo"), Literal(42))])
    assert parse("{foo}") == Object([(Literal("foo"), Index(Literal("foo")))])
    assert parse('{"foo"}') == Object([(Literal("foo"), Index(Literal("foo")))])
    assert parse('{["foo" + "bar"]: 42}') == Object(
        [(Add(String([Literal("foo")]), String([Literal("bar")])), Literal(42))]
    )
    assert parse('{"foo": [.[]]}') == Object([(Literal("foo"), Array([Iterator()]))])
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
    assert parse(".[] / 2") == Div(Iterator(), Literal(2))
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
