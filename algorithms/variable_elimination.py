from bayes_networks.bayes_network import BayesNet
from tools import normalize
from typing import List, Dict
import numpy as np
import itertools
from functools import reduce

class Factor:
    def __init__(self, vars_involved, cpt_f):
        self.vars_involved = vars_involved
        self.factor_func = cpt_f

    def __str__(self):
        return str(self.vars_involved)

    def involves(self, var):
        return var in self.vars_involved

    def multiply(self, funcs):
        def multiply_func(e):
            product = 1
            for f in funcs:
                product*= f(e)
            return product
        return multiply_func

    def marginalze(self, multiply_func, to_eliminate: str, to_eliminate_domain: List[str]):
        def marg_func(e_new):
            total = 0
            for value in to_eliminate_domain:
                e = {to_eliminate: value}
                total += multiply_func({**e, **e_new})
            return total
        return marg_func

    def merge_vars_involved(self, others, to_eliminate: str, to_eliminate_domain: List[str]):
        """ returns nodes involved in a new factor if the factors in self and others are merged """
        return (set(el for o in others for el in o.vars_involved).union(self.vars_involved)).difference({to_eliminate})

    def merge(self, others, to_eliminate: str, to_eliminate_domain: List[str]):
        involved = self.merge_vars_involved(others, to_eliminate, to_eliminate_domain)
        funcs = [self.factor_func] + [o.factor_func for o in others]

        return Factor(involved, self.marginalze(self.multiply(funcs), to_eliminate, to_eliminate_domain))

    def apply(self, ev):
        return self.factor_func(ev)

def _cpt_func(net: BayesNet, node: str, evidence: Dict[str, str]):
    def func(query: Dict[str, str]):
        combined = {**evidence, **query}
        if (node in combined and  (all(map(lambda x: x in combined.keys(), net.parents[node])))):
            return net.cpt_probability(node, combined)
        else:
            raise Exception('error', 'not valid cpt evl')

    return func

def _createFactor(net: BayesNet, node: str, evidence) -> Factor:
    return Factor(set(net.parents[node]).union({node}), _cpt_func(net, node, evidence))

def _count_new_fills(var, domain, factors, induced_edges):
    to_merge = [f for f in factors if f.involves(var)]

    if len(to_merge) >= 1:
        clique_nodes = (to_merge[0].merge_vars_involved(to_merge[1:], var, domain)).copy()
        clique_edges = set(itertools.combinations(sorted(clique_nodes), 2))
        return len(clique_edges.difference(induced_edges))

    return 0

def _min_fill_var(net, vars, factors: List[Factor]):
    """ Min-fill heuristic: Choose a variable that results in the smallest number of fill edges"""
    induced_edges = reduce \
        ( lambda acc, f: acc.union(set(itertools.combinations(sorted(f.vars_involved), 2)))
        , factors
        , set()
        )

    return np.argmin([_count_new_fills(v, net.domain[v], factors, induced_edges) for v in vars])

def infer(net: BayesNet, variable: str, evidence: Dict[str, str]):
    factors_left = []
    vars_left = []

    # initialize factors
    for node in net.ordered_vars:
        factor = _createFactor(net, node, evidence)
        factors_left.append(factor)
        if node!=variable and node not in evidence.keys():
            vars_left.append(node)

    while len(vars_left) > 0:
        node = vars_left.pop(_min_fill_var(net, vars_left, factors_left))
        to_merge = []
        to_pop = []
        for f_index, factor in enumerate(factors_left):
            if factor.involves(node):
                to_merge.append(factor)
                to_pop.append(f_index)

        while len(to_pop) != 0:
            factors_left.pop(to_pop.pop())

        if len(to_merge) >= 1:
            newfactor = to_merge[0].merge(to_merge[1:], node, net.domain[node])
            factors_left.append(newfactor)

    r = np.ones(len(net.domain[variable]))
    for i, f in enumerate(factors_left):
        for i, v in enumerate(net.domain[variable]):
            r[i] = r[i] * f.apply({variable: v})
    return normalize(r)
