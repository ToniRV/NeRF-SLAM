#!/usr/bin/env python3

class Key():
    def __init__(self, name: str, idx: int):
        self.name = name
        self.idx = idx

    def name(self):
        return self.name

    def idx(self):
        return self.idx

    def __hash__(self):
        return hash((self.name, self.idx))

    def __eq__(self, other):
        return (self.name, self.idx) == (other.name, other.idx)

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not(self == other)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"{self.name}_{self.idx}"
