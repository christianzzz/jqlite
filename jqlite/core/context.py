from typing import Optional, Type

from jqlite.core.filters import Fn


class Context:
    def __init__(self):
        self.dict = {}
        for fn in Fn.__subclasses__():
            self.dict[fn.name()] = fn

    def get(self, name: str) -> Optional[Type[Fn]]:
        return self.dict.get(name)
