#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This code evolves a known wave packet between two fixed positions
varying the system's length, the number of points, the time step,
and the Numerical Method, which could be: Pseud-Espectral,
Crank-Nicolson, or Runge-Kutta.
"""

# python standard
import os, time
from multiprocessing import Pool, TimeoutError
import logging
logger = logging.getLogger('wave-packet-logger')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(\
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# python extended
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

# our libraries
from lib.free_wave_packet import FreeWavePacket

# initialize particle and combinations of interest
methods = ['pe', 'cn', 'rk']
steps = [1e-20, 5e-20, 1e-19, 5e-19, 1e-18, 5e-18, \
    1e-17, 5e-17, 1e-16, 5e-16]
combinations = []

# generates a mesh of combinations of {method,L,N,dt}
for method in methods:
    for L in np.linspace(100,1000,7):
        for N in [2**n for n in range(8,13)]:
            for dt in steps:
                if dt < 1e-19 and method != 'pe':
                    continue
                combinations.append((method, L, N, dt))

def evolve_comb(combination):
    """
    This function apply the received parameter to a wave wacket
    evolving in time from -20 to +20 Angstrom
    Parameters
    ----------
    combination : tuple
        must have (method, L, N, dt)
    """
    try:
        method, L, N, dt = combination

        particle = FreeWavePacket()
        res = particle.numerical_evolution(L=L, N=N, dt=dt, \
            method=method)
        zf_real = res['zf_real']
        res_ana = particle.analytical_evolution(zf=zf_real)

        for k in res_ana.keys():
            res[k + '_ana'] = res_ana[k]

        message = "%s: L=%d, N=%d, dt=%.2e, " + \
                    "A/A0=%.5f, S=%.4f, G=%.4f, " + \
                    "A/A0_ana=%.5f, S_ana=%.4f, G_ana=%.4f, " + \
                    "time=%.5f"
        message = message % (method, L, N, dt,\
            res['a'], res['stdvar'], \
            res['skew'], res['a_real'], \
            res['stdvar_real'], res['skew_real'],
            res['program_time'])
        logger.info(message)

        return res

    except Exception as err:
        logger.error("Falha em %s: L=%d, N=%d, dt=%.2e" % \
            (method, L, N, dt))
        logger.error(str(err))
        return {}

pool = Pool(processes=8)
results = pd.DataFrame(pool.map(evolve_comb, combinations))

pec = results.loc[results['method'] == 'pe']
cnc = results.loc[results['method'] == 'cn']
rkc = results.loc[results['method'] == 'rk']

# stores data, since the process above might take even days
pec.to_csv('assets/free_wave_packet_results_pec.csv')
cnc.to_csv('assets/free_wave_packet_results_cnc.csv')
rkc.to_csv('assets/free_wave_packet_results_rkc.csv')

# loading
pec = pd.read_csv('assets/free_wave_packet_results_pec.csv')
cnc = pd.read_csv('assets/free_wave_packet_results_cnc.csv')
rkc = pd.read_csv('assets/free_wave_packet_results_rkc.csv')

# scale results for quality indicator
scaler = StandardScaler()
cols = ['stdvar', 'skew', 'a', 'stdvar_real', 'skew_real', 'a_real']
pec[cols] = scaler.fit_transform(pec[cols])
rkc[cols] = scaler.fit_transform(rkc[cols])
cnc[cols] = scaler.fit_transform(cnc[cols])

def minkowski(line, p=3):
    """
    calculates the minkowski distance of a given `line` that has
    at least columns [**stdvar**, **skew**, **a**] and 
    [**stdvar_real**, **skew_real**, **a_real**]

    Parameters
    ----------
    line : DataFrame row
        the row of a DataFrame
    p : int
        the p parameter of the Minkowski Distance
    
    Return
    ------
    dist : float
        the Minkowski Distance
    """
    x_num = [[line['stdvar'], line['skew'], \
        line['a']]]
    x_ana = [[line['stdvar_real'], line['skew_real'], \
        line['a_real']]]
    dist = cdist(XA=x_num, XB=x_ana, metric='minkowski', p=p)
    return dist[0][0]

pec['minkowski'] = pec.apply(minkowski, axis=1)
rkc['minkowski'] = rkc.apply(minkowski, axis=1)
cnc['minkowski'] = cnc.apply(minkowski, axis=1)

# stores the scaled data with the minkowsk distance
pec.to_csv('assets/free_wave_packet_results_pec_scaled.csv')
cnc.to_csv('assets/free_wave_packet_results_cnc_scaled.csv')
rkc.to_csv('assets/free_wave_packet_results_rkc_scaled.csv')