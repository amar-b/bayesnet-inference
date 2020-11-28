from bayes_networks.bayes_network import BayesNet
from tools import normalize
import numpy as np

def infer(net: BayesNet, variable: str, evidence, sample_size = 10000):
    """ Likelihood Weighting """
    non_evidence_vars = [v for v in net.ordered_vars if v not in evidence]

    def sample():
        weight = 1
        cur_state = {**evidence, **{v: "" for v in non_evidence_vars}}
        for var in net.ordered_vars:
            if var in evidence:
                weight *= net.cpt_probability(var, cur_state)
            else:
                cur_state[var] = net.cpt_sample(var, cur_state)
        return weight, cur_state[variable]

    weights = np.zeros(net.domain[variable].size)
    for count in range(sample_size):
        if (count%2000 == 0):
            print(f'{count}/{sample_size}')
        weight, sampled_value = sample()
        for value_index, value in enumerate(net.domain[variable]):
            if sampled_value == value:
                weights[value_index] += weight
    return normalize(weights)
