# jqlite

![PyPI](https://img.shields.io/pypi/v/jqlite)
![GitHub](https://img.shields.io/github/license/christianzzz/jqlite)
[![codecov](https://codecov.io/gh/christianzzz/jqlite/branch/develop/graph/badge.svg?token=9UE406IALD)](https://codecov.io/gh/christianzzz/jqlite)
[![Tests](https://github.com/christianzzz/jqlite/actions/workflows/tests.yml/badge.svg)](https://github.com/christianzzz/jqlite/actions/workflows/tests.yml)

An implementation of [jq](https://stedolan.github.io/jq/), the commandline JSON processor, for learning and fun.

## Installation

```shell
> pip install jqlite
```

## Examples:
```sh
> echo '{"foo": 0}' | jqlite
{
  "foo": 0
}

> echo '{"foo": [1, 2, 3, 4]}' | jqlite '[.foo | .[] | select(. % 2 == 0) | . * 2]'
[
  4.0,
  8.0
]
```