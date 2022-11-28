#!/usr/bin/env python3

from typing import List, Dict
import colored_glog as log
import torch as th

from factor_graph.key import Key

class Variable:
    def __init__(self, key: Key, value: th.Tensor):
        self.key = key
        self.value = value

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Variable(key={self.key}, value={self.value})"

class Variables:
    def __init__(self):
        self.vars: Dict[Key, Variable] = {}
        pass

    def add(self, variable: Variable):
        self.vars[variable.key] = variable

    # The order of the keys is important
    def at(self, keys: List[Key]) -> th.Tensor:
        # Stack all variables in the order of the keys
        return th.hstack([self.vars[key].value for key in keys])

    # The order of the keys is important
    def stack(self) -> th.Tensor:
        # Stack all variables in the order of the keys
        # TODO: super slow, we should keep an hstack perhaps
        return th.hstack(list(map(lambda x: x.value, self.vars.values())))

    # The order of the delta is important, it must match the order of the keys
    # in the Variables object self.vars
    # delta -> [B, N]
    # vars -> [B, N]
    def retract(self, delta: th.Tensor):
        log.check_eq(delta.shape[1], len(self.vars))
        # TODO
        # Retract variables in parallel, rather than sequentially
        # Use ordered dict, instead of retrieving in order
        x_new = {}
        for delta_i, key in zip(delta.t(), self.vars.keys()):
            x_new[key] = Variable(key, self.vars[key].value + delta_i.unsqueeze(0).t())
        return x_new

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Variables(vars={self.vars})"

