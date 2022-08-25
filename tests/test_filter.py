import pytest

from jqlite.core.filters import (
    Identity,
    Empty,
    Index,
    Literal,
    Semi,
    Array,
    Object,
    Pipe,
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Ne,
    Gt,
    Lt,
    Le,
    Ge,
    Sum,
    Length,
    Select,
    Mod,
    Map,
    String,
    Iteration,
    Slice,
    Range,
)


def test_identify():
    f = Identity()
    for v in [None, True, False, 3.14, "foo", [42], {"foo": "bar"}]:
        assert list(f.input(v)) == [v]


def test_iterator():
    assert list(Iteration().input([1, 2, 3])) == [1, 2, 3]
    assert list(Iteration().input({"foo": 1, "bar": 2})) == [1, 2]


def test_index():
    assert list(Index(Literal(1)).input([1, 2, 3])) == [2]
    assert list(Index(Literal("foo")).input({"foo": "bar"})) == ["bar"]


def test_slice():
    assert list(Slice([Literal(0), Literal(2)]).input([1, 2, 3])) == [[1, 2]]

    with pytest.raises(TypeError, match="Slice indices must be integers"):
        list(Slice([Literal("foo"), Literal("bar")]).input([1, 2, 3]))

    with pytest.raises(TypeError, match="Slice indices must be integers"):
        list(Slice([Literal(1.1), Literal(2.1)]).input([1, 2, 3]))

    assert list(Slice([Literal(0.0), Literal(2.0)]).input([1, 2, 3])) == [[1, 2]]


def test_literal():
    for v in [None, True, False, 3.14, "foo"]:
        f = Literal(v)
        assert list(f.input("whatever")) == [v]


def test_empty():
    f = Empty()
    assert list(f.input(42)) == []


def test_semi():
    f = Semi([Index(Literal("foo")), Index(Literal("bar"))])
    assert list(f.input({"foo": 1, "bar": 2})) == [1, 2]


def test_array():
    assert list(Array([]).input("whatever")) == [[]]

    f = Array([Index(Literal("foo")), Index(Literal("bar"))])
    assert list(f.input({"foo": 1, "bar": 2})) == [[1, 2]]

    assert list(Array([Iteration()]).input([1, 2, 3])) == [[1, 2, 3]]


def test_object():
    assert list(Object([]).input("whatever")) == [{}]

    f = Object(
        [
            (Literal("foo"), Index(Literal("foo"))),
            (Literal("bar"), Index(Literal("bar"))),
        ]
    )
    assert list(f.input({"foo": 1, "bar": 2, "baz": 3})) == [{"foo": 1, "bar": 2}]

    f = Object([(Literal("foo"), Iteration()), (Literal("bar"), Iteration())])
    assert list(f.input([1, 2])) == [
        {"foo": 1, "bar": 1},
        {"foo": 1, "bar": 2},
        {"foo": 2, "bar": 1},
        {"foo": 2, "bar": 2},
    ]

    f = Object(
        [
            (
                Pipe(
                    [
                        Array(
                            [
                                Literal("a"),
                                Literal("b"),
                            ]
                        ),
                        Iteration(),
                    ]
                ),
                Iteration(),
            ),
            (Literal("bar"), Iteration()),
        ]
    )
    assert list(f.input([1, 2])) == [
        {"a": 1, "bar": 1},
        {"a": 1, "bar": 2},
        {"a": 2, "bar": 1},
        {"a": 2, "bar": 2},
        {"b": 1, "bar": 1},
        {"b": 1, "bar": 2},
        {"b": 2, "bar": 1},
        {"b": 2, "bar": 2},
    ]


def test_string():
    f = String([Literal("abc")])
    assert list(f.input(None)) == ["abc"]

    f = String([Literal("abc"), Identity()])
    assert list(f.input("def")) == ["abcdef"]

    f = String([Literal("a"), Add(Literal(1), Literal(1)), Literal("c")])
    assert list(f.input(None)) == ["a2c"]

    f = String([Literal("abc"), Iteration()])
    assert list(f.input(["1", "2"])) == ["abc1", "abc2"]


def test_pipe():
    f = Pipe([Index(Literal("foo")), Iteration()])
    assert list(f.input({"foo": [1, 2, 3]})) == [1, 2, 3]


def test_add():
    f = Add(Identity(), Literal(2))
    assert list(f.input(1)) == [3]


def test_sub():
    f = Sub(Identity(), Literal(2))
    assert list(f.input(3)) == [1]


def test_mul():
    f = Mul(Identity(), Literal(2))
    assert list(f.input(2)) == [4]


def test_div():
    f = Div(Identity(), Literal(2))
    assert list(f.input(4)) == [2]


def test_mod():
    f = Mod(Identity(), Literal(2))
    assert list(f.input(3)) == [1]
    assert list(f.input(4)) == [0]


def test_eq():
    f = Eq(Identity(), Literal(2))
    assert list(f.input(2)) == [True]
    assert list(f.input(3)) == [False]


def test_ne():
    f = Ne(Identity(), Literal(2))
    assert list(f.input(2)) == [False]
    assert list(f.input(3)) == [True]


def test_gt():
    f = Gt(Identity(), Literal(2))
    assert list(f.input(3)) == [True]
    assert list(f.input(2)) == [False]
    assert list(f.input(1)) == [False]


def test_ge():
    f = Ge(Identity(), Literal(2))
    assert list(f.input(3)) == [True]
    assert list(f.input(2)) == [True]
    assert list(f.input(1)) == [False]


def test_lt():
    f = Lt(Identity(), Literal(2))
    assert list(f.input(3)) == [False]
    assert list(f.input(2)) == [False]
    assert list(f.input(1)) == [True]


def test_le():
    f = Le(Identity(), Literal(2))
    assert list(f.input(3)) == [False]
    assert list(f.input(2)) == [True]
    assert list(f.input(1)) == [True]


def test_sum():
    f = Sum()
    assert list(f.input([])) == [None]
    assert list(f.input([1, 2, 3])) == [6]
    assert list(f.input(["foo", "bar"])) == ["foobar"]
    assert list(f.input([{"a": 1}, {"a": 2, "b": 3}])) == [{"a": 2, "b": 3}]


def test_length():
    f = Length()
    assert list(f.input([1, 2])) == [2]
    assert list(f.input({"foo": "bar"})) == [1]
    for v in [None, True, False, 3.14]:
        with pytest.raises(TypeError):
            list(f.input(v))


def test_select():
    f = Pipe([Iteration(), Select(Eq(Mod(Identity(), Literal(2)), Literal(0)))])
    assert list(f.input([1, 2, 3, 4])) == [2, 4]


def test_map():
    f = Map(Mul(Identity(), Literal(2)))
    assert list(f.input([1, 2, 3])) == [[2, 4, 6]]
    assert list(f.input({"foo": "bar"})) == [["barbar"]]


def test_range():
    assert list(Range(Literal(2)).input(None)) == [0, 1]
    assert list(Range(Literal(1), Literal(3)).input(None)) == [1, 2]
    assert list(Range(Literal(1), Literal(5), Literal(2)).input(None)) == [1, 3]
    assert list(Range(Semi([Literal(2), Literal(3)])).input(None)) == [0, 1, 0, 1, 2]
