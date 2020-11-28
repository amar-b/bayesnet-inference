from numpy import random as rnd
from bayes_networks.bntypes import BnData
from tools import normalize, hash_prob, first_index, random_choice
from typing import Tuple, Dict, List
import itertools

class BayesNet:
    def __init__(self, bnData: BnData):
        self.domain, self.parents, self.cpt = bnData
        self.ordered_vars = self._order_vars()
        self.children = self._map_children()
        self.parent_values = self.ParentValueCombinations(self.ordered_vars, self.domain, self.parents)

    @staticmethod
    def empty():
        return BayesNet(({}, {}, {}))

    def __str__(self):
        """print network as a joint probability expression"""
        dist = ""
        for v in self.ordered_vars:
            if (self.parents[v]):
                dist += f"p({v} | {', '.join(self.parents[v])})"
            else:
                dist += f"p({v})"
        return dist

    def root_nodes(self):
        """get root nodes in network (variables with prior probabilities)"""
        return [i for i in self.ordered_vars if len(self.parents[i]) == 0]
        
    def _order_vars(self):
        """get nodes ordered based on dependency. Root nodes appear earlier since they don't depend on others"""
        ordered_vars = []
        unordered_vars = list(self.parents.keys())

        for v in unordered_vars:
            if len(self.parents[v]) == 0:
                ordered_vars.append(v)
        
        while len(ordered_vars) != len(unordered_vars):
            for v in unordered_vars:
                if v not in ordered_vars:
                    if all(map(lambda t: t in ordered_vars, self.parents[v])) == True:
                        ordered_vars.append(v)
        
        return ordered_vars
        
    def _map_children(self):
        """maps a node to its children"""
        children = {n:[] for n in self.ordered_vars}
        for child, parents in self.parents.items():
            for parent in parents:
                children[parent].append(child)
        return children
        
    def get_markov_blanket(self, variable):
        """get list of variables in the markov blanket of a variable"""
        blanket = []
        
        for n in self.parents[variable] + self.children[variable]:
            blanket.append(n)

        for c in self.children[variable]:
            for p in self.parents[c]:
                if p not in self.parents[variable]:
                    blanket.append(p)

        return blanket

    def cpt_table(self, variable, query):
        """probability table of a variable given parent values (int query)"""
        parent_vals = {p: query[p] for p in self.parents[variable]}
        return normalize(self.cpt[hash_prob(variable, parent_vals)][:])
    
    def cpt_using_indexes(self, variable, index_query):
        mapped_query = {v: self.domain[v][i] for v, i in index_query.items()}
        return self.cpt[hash_prob(variable, mapped_query)]

    def cpt_probability(self, variable, query):
        """Get the probability given parent values (int query)"""
        val_index = first_index(self.domain[variable], query[variable])
        return self.cpt_table(variable, query)[val_index]

    def cpt_sample(self, variable, query):
        """choose a value according based on probability given its parent values (in query)"""
        return random_choice(self.domain[variable], self.cpt_table(variable, query))[0]
        
    def joint_probability(self, query):
        """gets the join probability distribution of the network"""
        p = 1.0
        if type(query == dict) and all(map(lambda v: v in self.ordered_vars, query.keys())):
            for v in self.ordered_vars:
                p *= self.cpt_probability(v, query)
            return p
        return 0.0

    def mb_table(self, variable, state):
        """probability table of a variable given its markov blanket (in state)"""
        updated_state = {**state}
        table = self.cpt_table(variable, updated_state)
        for i, val in enumerate(self.domain[variable]):
            updated_state[variable]=val
            for c in  self.children[variable]:
                table[i] *= self.cpt_probability(c, updated_state)

        return normalize(table, force=True)

    def mb_sample(self, variable, state) -> str:
        """choose a value accounting to probability of a variable given its markov blanket in (state)"""
        return random_choice(self.domain[variable], self.mb_table(variable, state))[0]

    def mb_sample_index(self, variable, state) -> Tuple[str,int]:
        """choose a value accounting to probability of a variable given its markov blanket in (state)"""
        return random_choice(self.domain[variable], self.mb_table(variable, state))

    def uniform_sample(self, variable) -> str:
        """choose a value uniformly at random from a variable's domain"""
        return rnd.choice(a=self.domain[variable])

    def get_parent_combinations(self, variable):
        return self.parent_values.get_values(variable)

    class ParentValueCombinations:
        """ Maps each variable to a list of dicts where each dict is a combination of parent values"""
        def __init__(self, variables, domains, parents):
            combos = {}
            for i, v in enumerate(variables):
                psdm = [[(p, i) for i,v in enumerate(domains[p])] for p in parents[v]]
                combos[v] = list(map(lambda x:dict(x), itertools.product(*psdm)))
            self.combos = combos

        def get_values(self, variable) -> List[Dict[str, str]]:
            return self.combos[variable]