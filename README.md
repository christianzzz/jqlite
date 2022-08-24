# jqlite

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