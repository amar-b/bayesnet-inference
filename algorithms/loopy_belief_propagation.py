from tools import normalize, hash_prob, multiply
from bayes_networks.bayes_network import BayesNet
from copy import deepcopy
import numpy as np

def infer(net: BayesNet, variable: str, evidence):
    tables = LoopyBP(deepcopy(net)).propgate(evidence)
    return tables[variable]

class LoopyBP:
    pi_inbox = {}
    lambda_inbox = {}
    pi_values = {}
    lambda_values = {}
    instantiations = {}
    
    THRESHOLD = 10**(-10)
    LIMIT = 10**8

    def __init__(self, bayesNet: BayesNet):
        self.net = bayesNet

    def initialize(self, observed_evidence):
        self.instantiations = observed_evidence
        for n in self.net.ordered_vars:
            self.pi_inbox[n] = {}
            self.lambda_inbox[n] = {}
            self.pi_values[n] = normalize(np.ones(len(self.net.domain[n])))
            self.lambda_values[n] = normalize(np.ones(len(self.net.domain[n])))
            for p in self.net.parents[n]:
                self.pi_inbox[n][p] = normalize(np.ones(len(self.net.domain[p])))
            for c in self.net.children[n]:
                self.lambda_inbox[n][c] = normalize(np.ones(len(self.net.domain[n])))

    def propgate(self, observed_evidence):
        self.initialize(observed_evidence)
        old_belifs = {v:normalize(np.ones(len(self.net.domain[v]))) for v in self.net.ordered_vars}
        new_belifs = {v:normalize(np.zeros(len(self.net.domain[v]))) for v in self.net.ordered_vars}
        
        count = 0
        while self.has_belief_changed(old_belifs, new_belifs) and count < self.LIMIT:
            count += 1
            old_belifs = deepcopy(new_belifs)
            pi_values = deepcopy(self.pi_values)
            lambda_values = deepcopy(self.lambda_values)
            pi_inbox = deepcopy(self.pi_inbox)
            lambda_inbox = deepcopy(self.lambda_inbox)

            # combine messages and update belifs
            for n in self.net.ordered_vars:
                pi_values[n] = self.pi_value(n)
                lambda_values[n] = self.lambda_value(n)
                new_belifs[n] = self.belifs(lambda_values[n],  pi_values[n])

            self.pi_values = deepcopy(pi_values)
            self.lambda_values = deepcopy(lambda_values)

            # write new messages for next iteration
            for n in self.net.ordered_vars:
                for p in self.net.parents[n]:
                    lambda_inbox[p][n] = self.lambda_msg(n,p)
                for c in self.net.children[n]:
                    pi_inbox[c][n] = self.pi_msg(n,c)

            self.pi_inbox = deepcopy(pi_inbox)
            self.lambda_inbox = deepcopy(lambda_inbox)

        return new_belifs

    def belifs(self, lambda_values, pi_values):
        return normalize(lambda_values * pi_values)

    def has_belief_changed(self, old_belifs, new_belifs):
        for v in self.net.ordered_vars:
            if np.where(np.abs(old_belifs[v] - new_belifs[v]) > self.THRESHOLD, 1, 0).sum() > 0:
                return True

        return False

    def lambda_value(self, node):
        pi_array = multiply \
            ( self.lambda_inbox[node].values()
            , self.lambda_message_to_self(node)
            )
        
        return normalize(pi_array)

    def pi_value(self, node):        
        if len(self.net.parents[node]) == 0:
            return self.net.cpt[hash_prob(node)]

        pi_msgs_from_parents = self.pi_inbox[node]
        pi_array = np.zeros(len(self.net.domain[node]))

        for parent_values in self.net.get_parent_combinations(node):

            node_cpt = self.net.cpt_using_indexes(node, parent_values)

            inner_product = 1
            for parent, parent_value_index in parent_values.items():
                inner_product *= pi_msgs_from_parents[parent][parent_value_index]
            
            pi_array =  pi_array + (node_cpt * inner_product)

        return normalize(pi_array)
        
    def pi_msg(self, src, dtn):
        """The π message a node (src) sends to its child (dtn)"""
        lambda_msgs_product = multiply \
            ( [msg[1] for msg in self.lambda_inbox[src].items() if msg[0]!=dtn]
            , self.lambda_message_to_self(src)
            )

        return normalize(self.pi_values[src] * lambda_msgs_product)

    def lambda_msg(self, src, dtn):
        """The λ message a node (src) sends to its parent (dtn)"""
        pi_msgs_to_src = self.pi_inbox[src]
        lambda_vals_of_src = self.lambda_values[src]
        
        msg_array = np.ones(self.net.domain[dtn].size)

        for dtn_value_index, dtn_value in enumerate(self.net.domain[dtn]):
            outer_sum = 0

            for src_value_index, src_value in enumerate(self.net.domain[src]):
                src_lambda_value = lambda_vals_of_src[src_value_index]

                inner_sum = 0
                for parent_values in self.net.get_parent_combinations(src):
                    if parent_values[dtn] == dtn_value_index:
                        src_cpt = self.net.cpt_using_indexes(src, parent_values)
                        inner_product = 1
                        for src_parent, src_parent_val_index in parent_values.items():
                            if src_parent != dtn:
                                inner_product *= pi_msgs_to_src[src_parent][src_parent_val_index]
                        inner_sum += src_cpt[src_value_index] * inner_product

                outer_sum += src_lambda_value * inner_sum

            msg_array[dtn_value_index] = outer_sum

        return normalize(msg_array)

    def lambda_message_to_self(self, node):
        if node in self.instantiations:
            return np.where(self.net.domain[node] == self.instantiations[node], 1, 0)
        else:
            return np.ones(len(self.net.domain[node]))