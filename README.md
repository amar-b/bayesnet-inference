# 605.745 - Reasoning Under Uncertainty Project.
This repository contains the source code used for the 605.745 semester project. The goal of this project is to study the run time and quality of approximation of approximate inference algorithms when applied to medical diagnosis networks. The approximate algorithms implemented are Gibbs sampling, Likelihood weighting and Loopy belief propagation. The Variable elimination and enumeration exact algorithms are included in this repository as well. The entry point, `main.py`, executes experiments using the HEPAR2, PATHFINDER and the AQRM-DT networks. Each experiment measures the run time and KL-divergence of the 3 approximate methods and variable elimination.

## Getting started
```
python -m pip install -r requirements.txt
python main.py
```
