# Originally from https://github.com/ryan112358/private-pgm/blob/master/examples/privbayes.py
# Modified by Authors under the Apache 2.0 license
# Changes:
# remove all code except PrivBayes inference implementation
# stop printing column names in privbayes_inference 

import numpy as np
from mbi import Dataset, Factor, FactoredInference, mechanism
import pandas as pd
import itertools
import argparse

"""
This file implements PrivBayes, with and without our graphical-model based inference.
Zhang, Jun, Graham Cormode, Cecilia M. Procopiuc, Divesh Srivastava, and Xiaokui Xiao. "Privbayes: Private data release via bayesian networks." ACM Transactions on Database Systems (TODS) 42, no. 4 (2017): 25.
"""

def privbayes_inference(domain, measurements, total):
    synthetic = pd.DataFrame()

    _, y, _, proj = measurements[0]
    y = np.maximum(y, 0)
    y /= y.sum()
    col = proj[0]
    synthetic[col] = np.random.choice(domain[col], total, True, y)
        
    for _, y, _, proj in measurements[1:]:
        # find the CPT
        col, dep = proj[0], proj[1:]
        # print(col)
        y = np.maximum(y, 0)
        dom = domain.project(proj)
        cpt = Factor(dom, y.reshape(dom.shape))
        marg = cpt.project(dep)
        cpt /= marg
        cpt2 = np.moveaxis(cpt.project(proj).values, 0, -1)
        
        # sample current column
        synthetic[col] = 0
        rng = itertools.product(*[range(domain[a]) for a in dep])
        for v in rng:
            idx = (synthetic.loc[:,dep].values == np.array(v)).all(axis=1)
            p = cpt2[v].flatten()
            if p.sum() == 0:
                p = np.ones(p.size) / p.size
            n = domain[col]
            N = idx.sum()
            if N > 0:
                synthetic.loc[idx,col] = np.random.choice(n, N, True, p)

    return Dataset(synthetic, domain)

