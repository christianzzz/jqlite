import pytest

from jqlite.core.json_ops import (
    type_,
    Type,
    to_string,
    to_json,
    assert_type,
    set_by_path,
    Value,
    Path,
    is_type,
    is_null,
    is_boolean,
    is_number,
    is_string,
    is_array,
    is_object,
    iterate,
    index,
    slice_,
    add,
    sub,
)


def test_type():
    assert type_(None) == Type.NULL
    assert type_(True) == Type.BOOLEAN
    assert type_(42) == Type.NUMBER
    assert type_(3.14) == Type.NUMBER
    assert type_("foo") == Type.STRING
    assert type_(["foo", "bar"]) == Type.ARRAY
    assert type_({"foo": "bar"}) == Type.OBJECT


def test_is_type():
    assert is_type(42, Type.NUMBER)
    assert is_type("foo", Type.NUMBER, Type.STRING)
    assert not is_type("foo", Type.ARRAY)


def test_is_null():
    assert is_null(None)
    assert not is_null(42)


def test_is_boolean():
    assert is_boolean(True)
    assert is_boolean(False)
    assert not is_boolean(42)


def test_is_number():
    assert is_number(42)
    assert is_number(3.14)
    assert not is_number("foo")


def test_is_string():
    assert is_string("foo")
    assert not is_string(42)


def test_is_array():
    assert is_array([])
    assert not is_array({})


def test_is_object():
    assert is_object({})
    assert not is_object([])


def test_to_string():
    assert to_string(None) == "null"
    assert to_string(True) == "true"
    assert to_string(False) == "false"
    assert to_string(42) == "42"
    assert to_string(42.0) == "42"
    assert to_string(3.14) == "3.14"
    assert to_string("foo") == "foo"
    assert to_string(["foo", "bar"]) == '["foo","bar"]'
    assert to_string({"foo": "bar", "bar": "baz"}) == '{"foo":"bar","bar":"baz"}'


def test_to_json():
    assert to_json(None) == "null"
    assert to_json(True) == "true"
    assert to_json(False) == "false"
    assert to_json(42) == "42"
    assert to_json(42.0) == "42"
    assert to_json(3.14) == "3.14"
    assert to_json("foo") == '"foo"'
    assert to_json(["foo", "bar"]) == '["foo","bar"]'
    assert to_json({"foo": "bar", "bar": "baz"}) == '{"foo":"bar","bar":"baz"}'


def test_iterate():
    assert list(iterate([1, 2, 3])) == [1, 2, 3]
    assert list(iterate({"a": 1, "b": 2})) == [1, 2]
    for v in [None, True, 42, "foo"]:
        with pytest.raises(TypeError, match=f"Cannot iterate over {type_(v)}"):
            iterate(v)


def test_index():
    assert index([1, 2, 3], 1) == 2
    assert index([1, 2, 3], 4) is None
    assert index([1, 2, 3], -1) == 3
    assert index({"a": 1, "b": 2}, "a") == 1
    assert index({"a": 1, "b": 2}, "c") is None

    with pytest.raises(TypeError, match=f"Indices must be integers or strings"):
        assert index([1, 2, 3], 1.1)

    with pytest.raises(TypeError, match="Cannot index array with string"):
        assert index([1, 2, 3], "a")

    with pytest.raises(TypeError, match="Cannot index object with number"):
        assert index({"a": 1}, 1)

    with pytest.raises(TypeError, match="Cannot index number with string"):
        assert index(3, "a")


def test_slice():
    assert slice_([1, 2, 3], slice(1, 3)) == [2, 3]
    assert slice_([1, 2, 3], slice(None, None, 2)) == [1, 3]
    assert slice_([1, 2, 3], slice(-2, -1)) == [2]

    for v in [None, True, 42, "foo", {"a": 1}]:
        with pytest.raises(TypeError, match=f"Cannot index {type_(v)} with slice"):
            slice_(v, slice(1))


def test_add():
    for v in [None, True, 42, "foo", {"a": 1}]:
        assert add(None, v) == v
        assert add(v, None) == v

    assert add(1, 1) == 2
    assert add("a", "b") == "ab"
    assert add(["a"], ["b"]) == ["a", "b"]
    assert add({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    with pytest.raises(TypeError, match="Cannot add number and boolean"):
        assert add(1, True)

    with pytest.raises(TypeError, match="Cannot add boolean and boolean"):
        assert add(True, False)

    with pytest.raises(TypeError, match="Cannot add string and number"):
        assert add("a", 3)

    with pytest.raises(TypeError, match="Cannot add array and string"):
        assert add(["a"], "a")

    with pytest.raises(TypeError, match="Cannot add object and array"):
        assert add({"a": 1}, ["a"])


def test_sub():
    assert sub(1, 1) == 0
    assert sub([1, 2], [2]) == [1]

    with pytest.raises(TypeError, match="Cannot sub null and number"):
        assert sub(None, 1)

    with pytest.raises(TypeError, match="Cannot sub number and boolean"):
        assert sub(1, True)

    with pytest.raises(TypeError, match="Cannot sub boolean and boolean"):
        assert sub(True, False)

    with pytest.raises(TypeError, match="Cannot sub string and number"):
        assert sub("a", 3)

    with pytest.raises(TypeError, match="Cannot sub array and string"):
        assert sub(["a"], "a")

    with pytest.raises(TypeError, match="Cannot sub object and array"):
        assert sub({"a": 1}, ["a"])


def test_assert_type():
    assert_type(None, Type.NULL)
    assert_type(True, Type.BOOLEAN)
    assert_type(42, Type.NUMBER)
    assert_type(3.14, Type.NUMBER)
    assert_type("foo", Type.STRING)
    assert_type(["foo", "bar"], Type.ARRAY)
    assert_type({"foo": "bar"}, Type.OBJECT)

    assert_type(42, Type.NULL, Type.NUMBER)

    with pytest.raises(TypeError, match="Expected number, got null"):
        assert_type(None, Type.NUMBER)

    with pytest.raises(TypeError, match="Expected number or array, got null"):
        assert_type(None, Type.NUMBER, Type.ARRAY)

    with pytest.raises(TypeError, match="Expected number, array or object, got null"):
        assert_type(None, Type.NUMBER, Type.ARRAY, Type.OBJECT)

    with pytest.raises(TypeError, match=r"Expected \(\), got null"):
        assert_type(None)

    with pytest.raises(TypeError, match="some message"):
        assert_type(None, Type.NUMBER, message="some message")


@pytest.mark.parametrize(
    "val,path,new_val,expected",
    [
        ({"a": 1}, ["a"], 2, {"a": 2}),
        ({}, ["a"], 2, {"a": 2}),
        ({}, ["a", "b"], 2, {"a": {"b": 2}}),
        (None, ["a", "b"], 2, {"a": {"b": 2}}),
        ([1, 2, 3], [1], 4, [1, 4, 3]),
        ([], [2], 1, [None, None, 1]),
        (None, [2], 1, [None, None, 1]),
        (None, [0, "a"], 1, [{"a": 1}]),
        ({}, ["a", 0], 2, {"a": [2]}),
        ({}, ("a", 0), 2, {"a": [2]}),
    ],
)
def test_set_by_path(val: Value, path: Path, new_val: Value, expected: Value):
    updated = set_by_path(val, path, new_val)
    assert updated is not val and updated == expected


def test_set_by_path_invalid():
    with pytest.raises(TypeError, match="Cannot index number with string"):
        set_by_path(3, ["a"], 2)
    with pytest.raises(TypeError, match="Cannot index object with number"):
        set_by_path({}, [0], 2)
    with pytest.raises(TypeError, match="Cannot index number with number"):
        set_by_path(3, [0], 2)
    with pytest.raises(TypeError, match="Cannot index array with string"):
        set_by_path([], ["a"], 2)
    with pytest.raises(TypeError, match="Indices must be integers or strings"):
        set_by_path([], [None], 2)
    with pytest.raises(TypeError, match="Indices must be integers or strings"):
        set_by_path([], [1.2], 2)
    with pytest.raises(ValueError, match="Negative index out of range"):
        set_by_path([1], [-2], 2)
