from bayes_networks.bayes_network import BayesNet
from tools import bif_parse, qmr
from os import path
import json
import numpy as np

class MedicalDiagnosNetwork:
    def __init__(self):
        self.net = self.get_network()

    def data_path(self) -> str:
        return ""

    def get_network(self) -> BayesNet:
        fpath = self.data_path()

        if path.exists(fpath):
            return BayesNet(bif_parse(fpath))

        elif fpath == "":
            return BayesNet.empty()

        raise Exception('error', 'not valid file path')

    def create_random_instance(self, percentage_obs):
        diseases = self.get_diseases()
        non_diseases = [n for n in self.net.ordered_vars if n not in diseases]

        evidence = {}
        for node in non_diseases:
            sums = np.zeros(len(self.net.domain[node]))
            combo_cnt = 0
            for parent_values in self.net.get_parent_combinations(node):
                combo_cnt +=1
                sums = sums + self.net.cpt_using_indexes(node, parent_values)
            probable = self.net.domain[node][np.where(sums/combo_cnt > 0.1)]

            if np.random.random() <= percentage_obs:
                evidence[node] = np.random.choice(probable)
        
        return {
            "query": np.random.choice(diseases),
            "evidence":  evidence
        }

    def get_random_instances(self, count=2, percentage_obs=0.2):
        base_name = str(self.__class__.__name__)
        instances= []

        for c in range(count):
            name = f"{base_name}_{c}"
            fpath = f"data/instances/{name}.json"
            if path.exists(fpath):
                with open(fpath) as f:
                    instances.append((name, json.load(f)))
            else:
                instance = self.create_random_instance(percentage_obs)
                with open(fpath, 'w') as outfile:
                    json.dump(instance, outfile)
                instances.append((name, instance))
                
        return instances

    def get_diseases(self):
        return []

class ChildNetwork(MedicalDiagnosNetwork):
    def data_path(self):
        return 'data/networks/child.bif'
    
    def get_diseases(self):
        return ["Disease"]

class Hepar2Network(MedicalDiagnosNetwork):
    def data_path(self):
        return 'data/networks/hepar2.bif'

    def get_random_instances(self, count=2, percentage_obs=0.01):
        return super().get_random_instances(count=count, percentage_obs=percentage_obs)

    def get_diseases(self):
        return [
            "PBC",                    # Primary biliary cholangitis
            "ChHepatitis",            # Active/Chronic chronic hepatitis
            "Steatosis",              # Steatosis
            "Cirrhosis",              # Decompensate/Compensate Cirrhosis
            "RHepatitis",             # Reactive hepatitis
            "THepatitis",             # Toxic hepatitis
            "Hyperbilirubinemia"      # Hyperbilirubinemia
        ]

class PathfinderNetwork(MedicalDiagnosNetwork):
    def data_path(self):
        return 'data/networks/pathfinder.bif'
    
    def get_diseases(self):
        return [ "Fault" ]
        
    def get_random_instances(self, count=2, percentage_obs=0.05):
        return super().get_random_instances(count=count, percentage_obs=percentage_obs)

class QMRNetwork(MedicalDiagnosNetwork):
    D = 40

    def get_network(self) -> BayesNet:
        fpath = f"data/networks/qmr.json"
        r = qmr(d=self.D)
        with open(fpath, 'w') as outfile:
            json.dump(self.serialize(r), outfile)
        return BayesNet(r)

    def get_diseases(self):
        return list(map(lambda x: str(x), range(self.D)))

    def get_random_instances(self, count=2, percentage_obs=0.1):
        return super().get_random_instances(count=count, percentage_obs=percentage_obs)

    def serialize(self, data):
        tolist = lambda d: {k:list(v) for k,v in d.items()}
        return {
            "doms": tolist(data[0]),
            "pars": tolist(data[1]),
            "cpt": tolist(data[2])
        }

    def deserialize(self, data):
        toNp = lambda d: {k:np.array(v) for k,v in d.items()}
        return (toNp(data["doms"]), toNp(data["pars"]), toNp(data["cpt"]))