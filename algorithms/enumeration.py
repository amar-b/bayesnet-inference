from bayes_networks.bayes_network import BayesNet
from tools import normalize
from typing import List, Dict
import numpy as np

def infer(net: BayesNet, variable : str, evidence: Dict[str, str]):
    table = np.ones(len(net.domain[variable]))
    for index, val in enumerate(net.domain[variable]):
        updated_ev = {**evidence, variable:val}
        table[index] = _enumerate_ask(net, net.ordered_vars, updated_ev)
    return normalize(table)

def _enumerate_ask(net: BayesNet, ordered_vars: List[str], query: Dict[str, str]):
    if len(ordered_vars) == 0:
        return 1.0
    
    first_var = ordered_vars[0]
    if first_var in query:
        p_given_e = net.cpt_probability(first_var, query)
        return p_given_e * _enumerate_ask(net, ordered_vars[1:], query)
    else:
        factor = 0
        for val in net.domain[first_var]:
            updated_ev = {**query, first_var:val}
            p_given_e = net.cpt_probability(first_var, updated_ev)
            factor += p_given_e * _enumerate_ask(net, ordered_vars[1:], updated_ev)
        return factor

