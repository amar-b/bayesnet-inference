from bayes_networks.bayes_network import BayesNet
from tools import normalize
from numpy import random as rnd
import numpy as np
from functools import reduce

def infer(net: BayesNet, variable :str, evidence, burn_in = 2500, sample_size = 10000):
    """ Gibbs sampling """
    non_evidence_vars = [v for v in net.ordered_vars if v not in evidence]
    cur_state = {**evidence, **{v: rnd.choice(net.domain[v]) for v in non_evidence_vars}}

    counts_vector = np.zeros(net.domain[variable].size)
    chain_size = 0

    while chain_size < sample_size:
        if (chain_size%2000 == 0):
            print(f'{chain_size}/{sample_size+burn_in}')

        for ne_var in non_evidence_vars:
            val, idx = net.mb_sample_index(ne_var, cur_state)
            cur_state[ne_var] = val

            if chain_size > burn_in and variable == ne_var:
                counts_vector[idx] +=1

        chain_size +=1

    return normalize(counts_vector)
