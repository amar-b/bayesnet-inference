from functools import reduce
import re
import numpy as np
import random
import math
import itertools

""" Common Helpers """
def multiply(table, initial_value):
    try:
        return reduce(lambda acc, v: acc*v, table, initial_value)
    except:
        raise

def normalize(array: np.ndarray, force=False) -> np.ndarray:
    total = array.sum()
    
    if total != 0:
        return array/total
    elif force == True:
        return normalize(np.ones(array.size))
    else:
        return array

def random_choice(choices, distribution):
    smax = random.uniform(0, 1)
    s = 0

    for index, choice in enumerate(choices):
        s += distribution[index]
        if s >= smax:
            return (choice, index)
    raise Exception('error', 'invalid choic')

def uniform_choice(choices):
    return random_choice(choices, normalize(np.ones(len(choices))))

def hash_prob(variable, parent_values=None):
    if parent_values:
        return f"{variable}{(frozenset(parent_values.items()))}"
    else:
        return variable

def first_index(table, val):
    return np.nonzero(table==val)[0][0]

def laplace_smooth(table):
    small = 1
    large = 1e200
    return normalize(((table*large) + small)/(table.sum()*large + len(table)*small))

def kl_divergence_safe(act, approx):
    act_smooth = laplace_smooth(act)
    apx_smooth = laplace_smooth(approx)
    return (act_smooth*np.log2(act_smooth/apx_smooth)).sum()

""" Make QMR """
def noisy_or(n, n_range, leak_range):
    s = sum([random.uniform(n_range[0], n_range[1]) for _ in range(n)])
    return math.exp(-random.uniform(leak_range[0], leak_range[1])-s)

def qmr(d=100, f_ratio=8, d_deg=0.02):
    """ Create random qmr network"""
    f = d*f_ratio
    var_to_domain = {str(i): np.array(["1","0"]) for i in range(d+f)}
    var_to_parents = {str(i): [] for i in range(d+f)}
    var_to_cpt = {}
    for i in range(d):
        p = random.uniform(0,1)
        var_to_cpt[hash_prob(str(i))] = np.array([p, 1-p])
        for c in random.choices(list(range(d, d+f)), k=int(d_deg*f)):
            var_to_parents[str(c)].append(str(i))
    for i in range(d, d+f):
        if len(var_to_parents[str(i)]) == 0:
            var_to_parents[str(i)].append(str(random.choice(range(d))))
        var_to_parents[str(i)] = np.array(var_to_parents[str(i)])

    for v, ps in var_to_parents.items():
        if len(ps) != 0:
            psdm = [[(p, v) for v in ["1","0"]] for p in ps]
            for config in itertools.product(*psdm):
                p = noisy_or(len(ps), (0, 1), (0, 0.01))
                table = np.array([p, 1-p])
                var_to_cpt[hash_prob(str(v), dict(config))] = np.array(table)
    return (var_to_domain, var_to_parents, var_to_cpt)

""" Bif Parsing """
def read_csv_str(csv_row): 
    return list(map(lambda t: t.strip(), csv_row.split(", ")))

def read_csv_float(csv_row):
    return  list(map(lambda t: float(t.strip()), csv_row.split(", ")))

def bif_parse(filepath: str):
    file = open(filepath)

    nextline = lambda : file.readline().strip()

    name = re.compile(r"^variable (.+) \{$")
    domain = re.compile(r"^type discrete \[ \d+ \] \{ (.+) \};\s*$")
    mar_prob = re.compile(r"^probability \( ([a-zA-Z0-9_]+) \) \{$")
    cond_prob = re.compile(r"^probability \( ([a-zA-Z0-9_]+) \| (.+) \) \{$")
    mar_prob_val = re.compile(r"^table (.+);$")
    cond_prob_val = re.compile(r"^\((.+)\) (.+);$")

    variable_to_domain = {}
    variable_to_parents = {}
    variable_to_cpt = {}

    while True:
        line = nextline()
        if not line:
            break
        
        name_match = name.match(line)
        mar_prob_match = mar_prob.match(line)
        cond_prob_match = cond_prob.match(line)

        if (name_match):
            variable = name_match.group(1)
            variable_domain = read_csv_str(domain.match(nextline()).group(1))
            variable_to_domain[variable] = np.array(variable_domain)
        
        elif (mar_prob_match):
            variable = mar_prob_match.group(1)
            variable_to_parents[variable] = []
            table = read_csv_float(mar_prob_val.match(nextline()).group(1))
            variable_to_cpt[hash_prob(variable)] = np.array(table)
            
        elif (cond_prob_match):
            variable = cond_prob_match.group(1)
            parents = read_csv_str(cond_prob_match.group(2))
            variable_to_parents[variable] = parents
            while True:
                entry = cond_prob_val.match(nextline())
                if entry == None:
                    break
                parents_values = np.array(read_csv_str(entry.group(1)))
                table = read_csv_float(entry.group(2))
                mapping = dict(zip(parents, parents_values))
                variable_to_cpt[hash_prob(variable, mapping)] = np.array(table)

    return (variable_to_domain, variable_to_parents, variable_to_cpt)