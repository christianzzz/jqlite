from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    Union,
    Optional,
    Iterable,
    Sequence,
    List,
)

Value = Union[None, bool, int, float, str, list, dict]


class Type(Enum):
    NULL = "null"
    BOOLEAN = "boolean"
    NUMBER = "number"
    STRING = "string"
    ARRAY = "array"
    OBJECT = "object"

    def __str__(self):
        return self.value


PathIndex = Union[int, str]
Path = Sequence[PathIndex]


class BaseDelegate(ABC):
    @classmethod
    @abstractmethod
    def type_(cls) -> Type:
        ...

    @classmethod
    def to_string(cls, val: Value) -> str:
        return cls.to_json(val)

    @classmethod
    @abstractmethod
    def to_json(cls, val: Value) -> str:
        ...

    @classmethod
    def iterate(cls, val: Value) -> Iterable[Value]:
        raise TypeError(f"Cannot iterate over {type_(val)}")

    @classmethod
    def index(cls, val: Value, idx: Value) -> Optional[Value]:
        _raise_index_error(val, idx)

    @classmethod
    def add(cls, lval: Value, rval: Value) -> Value:
        raise TypeError(f"Cannot add {type_(lval)} and {type_(rval)}")

    @classmethod
    def sub(cls, lval: Value, rval: Value) -> Value:
        raise TypeError(f"Cannot sub {type_(lval)} and {type_(rval)}")

    @classmethod
    def set_by_path(cls, val: Value, path: Path, new_val: Value) -> Value:
        if not path:
            return new_val
        _raise_index_error(val, path[0])


class NullDelegate(BaseDelegate):
    @classmethod
    def type_(cls) -> Type:
        return Type.NULL

    @classmethod
    def to_json(cls, val: Value) -> str:
        return "null"

    @classmethod
    def add(cls, lval: Value, rval: Value) -> Value:
        return rval

    @classmethod
    def set_by_path(cls, val: Value, path: Path, new_val: Value) -> Value:
        if not path:
            return new_val

        if is_int(path[0]):
            return set_by_path([], path, new_val)
        elif is_string(path[0]):
            return set_by_path({}, path, new_val)
        else:
            _raise_index_error(val, path[0])


class BooleanDelegate(BaseDelegate):
    @classmethod
    def type_(cls) -> Type:
        return Type.BOOLEAN

    @classmethod
    def to_json(cls, val: Value) -> str:
        return "true" if val else "false"

    @classmethod
    def add(cls, lval: Value, rval: Value) -> Value:
        if is_null(rval):
            return lval
        raise TypeError(f"Cannot add {type_(lval)} and {type_(rval)}")


class NumberDelegate(BaseDelegate):
    @classmethod
    def type_(cls) -> Type:
        return Type.NUMBER

    @classmethod
    def to_json(cls, val: Value) -> str:
        if is_int(val):
            val = int(val)
        return str(val)

    @classmethod
    def _isint(cls, val: Value):
        return isinstance(val, int) or (isinstance(val, float) and val == int(val))

    @classmethod
    def add(cls, lval: Value, rval: Value) -> Value:
        if is_null(rval):
            return lval
        elif is_number(rval):
            return lval + rval
        else:
            raise TypeError(f"Cannot add {type_(lval)} and {type_(rval)}")

    @classmethod
    def sub(cls, lval: Value, rval: Value) -> Value:
        if is_number(rval):
            return lval - rval
        raise TypeError(f"Cannot sub {type_(lval)} and {type_(rval)}")


class StringDelegate(BaseDelegate):
    @classmethod
    def type_(cls) -> Type:
        return Type.STRING

    @classmethod
    def to_string(cls, val: Value) -> str:
        return val

    @classmethod
    def to_json(cls, val: Value) -> str:
        return f'"{val}"'

    @classmethod
    def add(cls, lval: Value, rval: Value) -> Value:
        if is_null(rval):
            return lval
        elif is_string(rval):
            return lval + rval
        else:
            raise TypeError(f"Cannot add {type_(lval)} and {type_(rval)}")


class ArrayDelegate(BaseDelegate):
    @classmethod
    def type_(cls) -> Type:
        return Type.ARRAY

    @classmethod
    def to_json(cls, val: Value) -> str:
        return "[" + ",".join(to_json(v) for v in val) + "]"

    @classmethod
    def iterate(cls, val: Value) -> Iterable[Value]:
        return val

    @classmethod
    def index(cls, val: Value, idx: Value) -> Optional[Value]:
        if is_int(idx):
            try:
                return val[int(idx)]
            except IndexError:
                return None
        else:
            _raise_index_error(val, idx)

    @classmethod
    def add(cls, lval: Value, rval: Value) -> Value:
        if is_null(rval):
            return lval
        elif is_array(rval):
            return lval + rval
        else:
            raise TypeError(f"Cannot add {type_(lval)} and {type_(rval)}")

    @classmethod
    def sub(cls, lval: Value, rval: Value) -> Value:
        if is_array(rval):
            # TODO: Performance improvement, requires lists and objects to be hashable,
            #  which is a big change to the underlying data models.
            return [v for v in lval if v not in rval]
        raise TypeError(f"Cannot sub {type_(lval)} and {type_(rval)}")

    @classmethod
    def set_by_path(cls, val: Value, path: Path, new_val: Value) -> Value:
        if not path:
            return new_val

        [path_index, *rest] = path

        if is_int(path_index):
            if path_index < -len(val):
                raise ValueError("Negative index out of range")
            val = list(val)
            while len(val) <= path_index:
                val.append(None)
            val[path_index] = set_by_path(val[path_index], rest, new_val)
            return val
        else:
            _raise_index_error(val, path_index)


class ObjectDelegate(BaseDelegate):
    @classmethod
    def type_(cls) -> Type:
        return Type.OBJECT

    @classmethod
    def to_json(cls, val: Value) -> str:
        return (
            "{" + ",".join(f"{to_json(k)}:{to_json(v)}" for k, v in val.items()) + "}"
        )

    @classmethod
    def iterate(cls, val: Value) -> Iterable[Value]:
        return val.values()

    @classmethod
    def index(cls, val: Value, idx: Value) -> Optional[Value]:
        if is_string(idx):
            return val.get(idx)
        else:
            _raise_index_error(val, idx)

    @classmethod
    def add(cls, lval: Value, rval: Value) -> Value:
        if is_null(rval):
            return lval
        elif is_object(rval):
            return {**lval, **rval}
        else:
            raise TypeError(f"Cannot add {type_(lval)} and {type_(rval)}")

    @classmethod
    def set_by_path(cls, val: Value, path: Path, new_val: Value) -> Value:
        if not path:
            return new_val

        [path_index, *rest] = path

        if is_string(path_index):
            val = dict(val)
            val[path_index] = set_by_path(val.get(path_index), rest, new_val)
            return val
        else:
            _raise_index_error(val, path_index)


_DELEGATE_DICT = {
    type(None): NullDelegate(),
    bool: BooleanDelegate,
    int: NumberDelegate,
    float: NumberDelegate,
    str: StringDelegate,
    list: ArrayDelegate,
    dict: ObjectDelegate,
}


def _get_delegate(val: Union[Value, Value]) -> BaseDelegate:
    return _DELEGATE_DICT[type(val)]


def type_(val: Value) -> Type:
    return _get_delegate(val).type_()


def is_type(val: Value, *types: Type) -> bool:
    return type_(val) in types


def is_null(val: Value) -> bool:
    return val is None


def is_boolean(val: Value) -> bool:
    return isinstance(val, bool)


def is_number(val: Value) -> bool:
    return isinstance(val, (int, float)) and not is_boolean(val)


def is_int(val: Value) -> bool:
    return isinstance(val, int) or (isinstance(val, float) and val == int(val))


def is_string(val: Value) -> bool:
    return isinstance(val, str)


def is_array(val: Value) -> bool:
    return isinstance(val, list)


def is_object(val: Value) -> bool:
    return isinstance(val, dict)


def to_string(val: Value) -> str:
    return _get_delegate(val).to_string(val)


def to_json(val: Value) -> str:
    return _get_delegate(val).to_json(val)


def iterate(val: Value) -> Iterable[Value]:
    return _get_delegate(val).iterate(val)


def index(val: Value, idx: Value) -> Optional[Value]:
    return _get_delegate(val).index(val, idx)


def slice_(val: Value, s: slice) -> List[Value]:
    if isinstance(val, list):
        return val[s]
    else:
        raise TypeError(f"Cannot index {type_(val)} with slice")


def add(lval: Value, rval: Value) -> Value:
    return _get_delegate(lval).add(lval, rval)


def sub(lval: Value, rval: Value) -> Value:
    return _get_delegate(lval).sub(lval, rval)


def assert_type(val: Value, *types: Type, message: Optional[str] = None):
    if not is_type(val, *types):
        raise TypeError(message or _default_assert_type_message(val, *types))


def _default_assert_type_message(val: Value, *types: Type) -> str:
    if len(types) == 0:
        type_str = str(types)
    elif len(types) == 1:
        type_str = str(types[0])
    else:
        [*leading, last] = types
        type_str = f'{", ".join(str(x) for x in leading)} or {last}'

    return f"Expected {type_str}, got {type_(val)}"


def set_by_path(val: Value, path: Path, new_val: Value) -> Value:
    return _get_delegate(val).set_by_path(val, path, new_val)


def _raise_index_error(val: Value, path_index: PathIndex):
    if not is_int(path_index) and not is_string(path_index):
        raise TypeError(f"Indices must be integers or strings")
    raise TypeError(f"Cannot index {type_(val)} with {type_(path_index)}")
