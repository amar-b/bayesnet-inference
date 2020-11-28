from typing import Tuple, Dict, List
from numpy import ndarray


"""BnDomains represents the variable domains for  all the variables in a bayes network 
    - Internally BnDomains is a dictionary mapping $key to $Value
    - $key: A str representing the variable name
    - $Value: A np.ndarray of np.str_ representing the domain values
"""
BnDomains = Dict[str, ndarray]



"""BnDomains represents directed edges of a bayes net
    - Internally BnParents maps a child to a list of parents
    - $key: A a str representing the variable name
    - $Value: A list of str representing the parent variable names
"""
BnParents = Dict[str, List[str]]



"""CptTables Type represents the CPT tables for all the variables in a bayes network 
    - Internally CptTables is a dictionary mapping $key to $Value
    - $key:
        - For root nodes this is the variable name
        - for non-root nodes this is the variable name concatenated with a hash
            + hash encodes the parent values for a cpt entry
    - $Value: A np.ndarray of np.float64 representing the probabilities
"""
BnCptTables = Dict[str, ndarray]



"""BayesNetPyData represents the data in a bayes network
    - $domains is a dictionary mapping a variable to a list representing its domain
    - CptTables is defined above
"""
BnData = Tuple[BnDomains, BnParents, BnCptTables]