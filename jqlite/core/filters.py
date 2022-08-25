import itertools
import operator
from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple

from more_itertools import chunked

from jqlite.core.json_ops import (
    iterate,
    index,
    slice_,
    Value,
    type_,
    is_int,
    to_string,
    add,
    sub,
)


class Filter(ABC):
    @abstractmethod
    def input(self, val: Value) -> Iterable[Value]:
        ...

    def __eq__(self, other) -> bool:
        return self.__class__ == other.__class__


class Identity(Filter):
    def input(self, val: Value) -> Iterable[Value]:
        yield val

    def __str__(self):
        return "."

    def __repr__(self):
        return "Identity()"


class Iteration(Filter):
    def input(self, val: Value) -> Iterable[Value]:
        yield from iterate(val)

    def __str__(self):
        return ".[]"

    def __repr__(self):
        return "Iteration()"


class Index(Filter):
    def __init__(self, filter: Filter):
        self.filter = filter

    def input(self, val: Value) -> Iterable[Value]:
        for idx in self.filter.input(val):
            yield index(val, idx)

    def __str__(self):
        return f".[{self.filter}]"

    def __repr__(self):
        return f"Index({repr(self.filter)})"


class Slice(Filter):
    def __init__(self, filters: Iterable[Filter]):
        self.filters = filters

    def input(self, val: Value) -> Iterable[Value]:
        for idx in itertools.product(*(f.input(val) for f in self.filters)):
            indices = []
            for i in idx:
                if i is None:
                    indices.append(None)
                elif is_int(i):
                    indices.append(int(i))
                else:
                    raise TypeError("Slice indices must be integers")
            yield slice_(val, slice(*indices))

    def __eq__(self, other):
        return super().__eq__(other) and self.filters == other.filters

    def __str__(self):
        return ".[" + ":".join(str(f) for f in self.filters) + "]"

    def __repr__(self):
        return f'Slice([{", ".join(repr(f) for f in self.filters)}])'


class Literal(Filter):
    def __init__(self, literal: Value):
        self.literal = literal

    def input(self, _: Value) -> Iterable[Value]:
        yield self.literal

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and self.literal == other.literal

    def __str__(self):
        return str(self.literal)

    def __repr__(self):
        return f"Literal({self.literal})"


class Semi(Filter):
    def __init__(self, filters: List[Filter]):
        self.filters = filters

    def input(self, val: Value) -> Iterable[Value]:
        for f in self.filters:
            for v in f.input(val):
                yield v

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and self.filters == other.filters

    def __str__(self):
        return ";".join(str(f) for f in self.filters)

    def __repr__(self):
        return f"Semi({self.filters})"


class Array(Filter):
    def __init__(self, filters: List[Filter]):
        self.filters = filters

    def input(self, val: Value) -> Iterable[Value]:
        result = []
        for f in self.filters:
            for v in f.input(val):
                result.append(v)
        yield result

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and self.filters == other.filters

    def __str__(self):
        return "[" + ",".join(str(f) for f in self.filters) + "]"

    def __repr__(self):
        return "Array(" + ",".join(repr(f) for f in self.filters) + ")"


class Object(Filter):
    def __init__(self, filter_pairs: List[Tuple[Filter, Filter]]):
        self.filter_pairs = filter_pairs

    def input(self, val: Value) -> Iterable[Value]:
        filters = []
        for k, v in self.filter_pairs:
            filters.append(k.input(val))
            filters.append(v.input(val))
        for x in itertools.product(*filters):
            yield {k: v for k, v in chunked(x, 2)}

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and self.filter_pairs == other.filter_pairs

    def __str__(self):
        return "{" + ",".join(f"{k}: {v}" for k, v in self.filter_pairs) + "}"

    def __repr__(self):
        return f"Object({self.filter_pairs!r})"


class String(Filter):
    def __init__(self, filters: List[Filter]):
        self.filters = filters

    def input(self, val: Value) -> Iterable[Value]:
        iterables = [f.input(val) for f in self.filters]
        for parts in itertools.product(*iterables):
            yield "".join(list(to_string(x) for x in parts))

    def __eq__(self, other):
        return super(String, self).__eq__(other) and self.filters == other.filters


class Pipe(Filter):
    def __init__(self, filters: List[Filter]):
        self.filters = filters

    def input(self, val: Value) -> Iterable[Value]:
        yield from self._input_with_filters(val, self.filters)

    def _input_with_filters(self, val: Value, filters: List[Filter]) -> Iterable[Value]:
        if not filters:
            yield val
            return
        [first, *rest] = filters
        for v in first.input(val):
            yield from self._input_with_filters(v, rest)

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and self.filters == other.filters

    def __str__(self):
        return " | ".join(str(f) for f in self.filters)

    def __repr__(self):
        return f"Pipe({self.filters!r})"


class Op:
    def __init__(self, op, sym: str):
        self.op = op
        self.sym = sym

    def __call__(self, *args, **kwargs):
        return self.op.__call__(*args, **kwargs)

    def __eq__(self, other) -> bool:
        return self.op is other.op

    def __str__(self):
        return self.sym


class BinOp(Filter):
    def __init__(self, left: Filter, right: Filter, op: Op):
        self.left = left
        self.right = right
        self.op = op

    def input(self, val: Value) -> Iterable[Value]:
        for v1 in self.left.input(val):
            for v2 in self.right.input(val):
                yield self.op(v1, v2)

    def __eq__(self, other) -> bool:
        return (
            super().__eq__(other)
            and self.left == other.left
            and self.right == other.right
            and self.op == other.op
        )

    def __str__(self):
        return f"{self.left} {self.op} {self.right}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.left!r}, {self.right!r})"


def make_bin_op(name: str, op):
    def constructor(self, left: Filter, right: Filter):
        BinOp.__init__(self, left, right, op)

    return type(name, (BinOp,), {"__init__": constructor})


Add = make_bin_op("Add", Op(add, "+"))
Sub = make_bin_op("Sub", Op(sub, "-"))
Mul = make_bin_op("Mul", Op(operator.mul, "*"))
Div = make_bin_op("Div", Op(operator.truediv, "/"))
Mod = make_bin_op("Mod", Op(operator.mod, "%"))
Eq = make_bin_op("Eq", Op(operator.eq, "=="))
Ne = make_bin_op("Ne", Op(operator.ne, "!="))
Gt = make_bin_op("Gt", Op(operator.gt, ">"))
Ge = make_bin_op("Ge", Op(operator.ge, ">="))
Lt = make_bin_op("Lt", Op(operator.lt, "<"))
Le = make_bin_op("Le", Op(operator.le, "<="))
And = make_bin_op("And", Op(lambda a, b: a and b, "and"))
Or = make_bin_op("Or", Op(lambda a, b: a or b, "or"))


class UnaryOp(Filter, ABC):
    def __init__(self, filter: Filter, op: Op):
        self.filter = filter
        self.op = op

    def input(self, val: Value) -> Iterable[Value]:
        for v in self.filter.input(val):
            yield self.op(v)


def make_unary_op(name: str, op):
    def constructor(self, left: Filter):
        UnaryOp.__init__(self, left, op)

    return type(name, (UnaryOp,), {"__init__": constructor})


Neg = make_unary_op("Neg", Op(operator.neg, "-"))
Pos = make_unary_op("Pos", Op(operator.pos, "+"))
Not = make_unary_op("Not", Op(operator.not_, "not"))


class Fn(Filter, ABC):
    @classmethod
    def name(cls) -> str:
        return cls.__name__.lower()


class Sum(Fn):
    def input(self, val: Value) -> Iterable[Value]:
        result = None
        for v in Iteration().input(val):
            result = add(result, v)
        yield result

    def __str__(self):
        return "sum"


class Length(Fn):
    def input(self, val: Value) -> Iterable[Value]:
        if isinstance(val, (list, dict, str)):
            yield len(val)
        else:
            raise TypeError(f"{type(val)} {val} has no length")

    def __str__(self):
        return "length"


class Select(Fn):
    def __init__(self, filter: Filter):
        self.filter = filter

    def input(self, val: Value) -> Iterable[Value]:
        for v in self.filter.input(val):
            if v is not None and v is not False:
                yield val

    def __str__(self):
        return f"select({self.filter})"


class Map(Fn):
    def __init__(self, filter: Filter):
        self.filter = filter

    def input(self, val: Value) -> Iterable[Value]:
        yield from Array([Pipe([Iteration(), self.filter])]).input(val)

    def __str__(self):
        return f"map({self.filter})"


class Range(Fn):
    def __init__(self, *args):
        self.start = Literal(0)
        self.step = Literal(1)

        if len(args) == 1:
            self.stop = args[0]
        elif len(args) == 2:
            self.start = args[0]
            self.stop = args[1]
        elif len(args) == 3:
            self.start = args[0]
            self.stop = args[1]
            self.step = args[2]
        else:
            raise ValueError(f"Wrong number of arguments: {len(args)}")

    def input(self, val: Value) -> Iterable[Value]:
        for start, stop, step in itertools.product(
            self.start.input(val), self.stop.input(val), self.step.input(val)
        ):
            while start < stop:
                yield start
                start += step

    def __str__(self):
        filters = []
        if self.start:
            filters.append(self.start)
        if self.stop:
            filters.append(self.stop)
        if self.step:
            filters.append(self.step)
        return f"range({'; '.join(str(f) for f in filters)})"


class Join(Fn):
    def __init__(self, filter: Filter):
        self.filter = filter

    def input(self, val: Value) -> Iterable[Value]:
        for sep in self.filter.input(val):
            yield sep.join(val)

    def __str__(self):
        return f"join({self.filter})"


class Type(Fn):
    def input(self, val: Value) -> Iterable[Value]:
        yield type_(val)


class Min(Fn):
    def input(self, val: Value) -> Iterable[Value]:
        if not val:
            yield None
        else:
            yield min(val)

    def __str__(self):
        return "min"


class Max(Fn):
    def input(self, val: Value) -> Iterable[Value]:
        if not val:
            yield None
        else:
            yield max(val)

    def __str__(self):
        return "max"


class Empty(Fn):
    def input(self, _: Value) -> Iterable[Value]:
        yield from ()

    def __str__(self):
        return "empty"

    def __repr__(self):
        return "Empty()"
