import numpy as np
import time
import csv
from os import path

from tools import kl_divergence_safe
from bayes_networks import medical_networks
from algorithms import loopy_belief_propagation
from algorithms import variable_elimination
from algorithms import likelihood_weighting
from algorithms import gibbs_sampling

def results_file_headers():
    return ["algorithm_name", "time", "kl_divergence"]

def try_create_results_file(fpath):
    if path.exists(fpath) == False:
        with open(fpath, 'w') as f:
            csv.writer(f).writerow(results_file_headers())
            return True
    return False

def write_to_results_file(fpath, row):
    with open(fpath, 'a') as f:
        csv.writer(f).writerow(row)

def run_inference_method(instance_name, algorithm_name, domain_size, infer_function, kl=None, count=1, sample_sizes=None):
    if sample_sizes:
        for sample_size in sample_sizes:
            _n = f"{algorithm_name}_{sample_size}"
            _f = lambda: infer_function(sample_size)
            run_inference_method(instance_name, _n, domain_size, _f, kl, count)

    else:
        fpath = f"data/results/{instance_name}.csv"
        print(instance_name)

        try_create_results_file(fpath)
        probabilities = np.zeros(shape=(count, domain_size))
        times = np.zeros(shape=count)

        for c in range(count):
            start_time = time.time()
            probabilities[c] = infer_function()
            times[c] = time.time() - start_time

        distribution = np.mean(probabilities, axis=0)
        results_row = [algorithm_name, np.mean(times), kl(distribution)] + list(distribution)
        write_to_results_file(fpath, results_row)

        return distribution

def run_experiment_for_dataset(data: medical_networks.MedicalDiagnosNetwork):
    for data_name, instance in data.get_random_instances(count=3):
        # initialize problem instance 
        net = data.net
        query_variable = instance['query']
        evidence = instance['evidence']
        domain_size = len(net.domain[query_variable])

        # set up inference methods
        ve = lambda: variable_elimination.infer(\
            net, query_variable, evidence)

        gbs = lambda t: gibbs_sampling.infer(\
            net, query_variable, evidence, sample_size=t)
        
        lw =  lambda t: likelihood_weighting.infer(\
            net, query_variable, evidence, sample_size=t)
        
        lbp = lambda: loopy_belief_propagation.infer(\
            net, query_variable, evidence)
        
        # run exact inference 
        p = run_inference_method(data_name,   "variable_elimination",     domain_size, ve,  lambda z: 0)

        # run approximate inference
        kl = lambda q: kl_divergence_safe(p, q)
        sample_sizes = [1000, 10000, 100000]
        trials = 5

        q_3 = run_inference_method(data_name, "loopy_belief_propagation", domain_size, lbp, kl)
        q_2 = run_inference_method(data_name, "likelihood_weighting",     domain_size, lw,  kl, trials, sample_sizes)
        q_1 = run_inference_method(data_name, "gibbs_sampling",           domain_size, gbs, kl, trials, sample_sizes)

if __name__ == "__main__":
    run_experiment_for_dataset(medical_networks.Hepar2Network())
    run_experiment_for_dataset(medical_networks.PathfinderNetwork())
    run_experiment_for_dataset(medical_networks.QMRNetwork())