#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains classes for simalating quantum heterostructures
"""

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.integrate import simps
import scipy.constants as cte
import scipy.special as sp
from scipy.signal import gaussian
from scipy.fftpack import fft, ifft, fftfreq
from datetime import datetime
from types import *

import os, time, re
from multiprocessing import Pool, TimeoutError

from band_structure_database import Alloy, Database

# very default values
DEFAULT_DT = 1e-18 # seconds

class HeteroStructure(object):
    """
    This class provides a basic framework for simulating an heterostructure
    based on its building properties: layers composition, materials used,
    sizes, room temperature

    It is possible to calculate:
    - eigenvalues and eigenfunctions
    - evolve waves in time under potential influence
    - calculate properties like photocurrent, transmission, reflextion

    The default potential is a Quantum Harmoic Oscillator for a wave length
    of 8.1 µm

    The inputs must always be in:
    - Energy: `eV`
    - Length: `nm`
    - Time: `s`
    """

    def __init__(self, N=8192):
        """
        Parameters
        ----------
        N : int
            the number of grid points, it should be a power of 2

        Examples
        --------
        >>> from heterostructures import HeteroStructure
        >>> device = HeteroStructure(1024)
        """

        # default grid size
        self.N = N
        self.hN = int(self.N/2) # half grid
        self.points_before = int(self.N/2)
        self.points_after = int(self.N/2)

        # useful atomic unities
        self.au_l = cte.value('atomic unit of length')
        self.au_t = cte.value('atomic unit of time')
        self.au_e = cte.value('atomic unit of energy')
        self.au_v = cte.value('atomic unit of electric potential')
        self.hbar_au = 1.0
        self.me_au = 1.0

        # other useful constants
        self.ev = cte.value('electron volt')
        self.c = cte.value('speed of light in vacuum') # m/s
        self.me = cte.value('electron mass')
        self.q = cte.value('elementary charge')
        self.au2ev = self.au_e / self.ev

        # specific for default quantum harmonic oscillator
        self.l = 0.0000081 # m
        self.f = self.c / self.l # Hz
        self.w = 2.0 * np.pi * self.f
        self.x_m = np.linspace(-5e-9,5e-9, self.N)
        self.v_j = 0.5 * self.me * self.x_m**2 * self.w**2
        self.x_nm = self.x_m / 1e-9
        self.v_ev = self.v_j / self.ev
        self.m_eff = np.ones(self.x_nm.size)

        # set default time increment
        self._set_dt()

        # adjust grid
        self.normalize_device()

    # getters and setters

    def get_system_state(self, n):
        """
        get a tuple with the nth eigenstate and corresponding eigenvalue

        Parameters
        ----------
        n : integer
            the number of the desired eigenstate/eigenvalue
        
        Returns
        -------
        info : tuple(DataFrame, float)
            a tuple consisting of a DataFrame with the eigenstate and a float
            with the eigenvalue
        """
        return (self.device[['x_nm','state_{}'.format(n)]], self.values[n])
    
    def get_system_states(self):
        """
        get a tuple with all the eigenstates and eigenvalues

        Returns
        -------
        info : tuple(DataFrame, array_like)
            a tuple consisting of a DataFrame with the eigenstates and an array
            of float with the corresponding eigenstates
        """
        working = self._working_names()
        return (self.device[['x_nm']+working], self.values)

    # operations

    def time_evolution(self, steps=2000, t0=0.0, \
        dt=None, imaginary=False, n=3, save=True, load=True):
        """
        This function will evolve the `system_waves` in time. It time is
        `imaginary`, then it will try to calculate the `n` first eigenvalues
        and eigenstates of the system.

        Parameters
        ----------
        steps : integer
            the number of time steps to evolve the system
        t0 : float
            if the start time, useful when working with time dependent 
            potentials, it must be in seconds
        dt : float
            the increment in time in **seconds** for each step, the default is 
            the system's default `dt` which is DEFAULT_DT
        imaginary : boolean
            *False* stands for not imaginary time evolution, while *True* stands
            for the opposite
        n : integer
            the number of eigenvalues and eigenstates to be calculated in case
            of imaginary time evolution, the edfault is `3`
        save : boolean
            whether to save the results of an imaginary evolution, which means
            save the eigenstates for further using. The eigenstates will be
            saved in a file at the folder `eigenfunction`
        load : boolean
            use stored eigenstates when available
        Returns
        -------
        self : GenericPotential
            the current GenericPotential object for further use in chain calls
        """
        
        self._set_dt(dt)
        t0_au = t0 / self.au_t
        
        if imaginary:

            try:
                fst = self.bias_raw
            except:
                fst = 0.0

            try:
                fdyn = self.ep_dyn_raw
            except:
                fdyn = 0.0

            filename = "{cn}_{n}_{sts}_{bias:.2f}_{dyn:.2f}.csv".format( \
                cn=self.__class__.__name__, n=n, sts=steps, bias=fst, dyn=fdyn)
            directory = "devices"
            full_filename = os.path.join(directory, filename)

            if load:
                try:
                    device = pd.read_csv(full_filename)
                    complex_cols = [c for c in device.columns \
                        if re.match('^.*_\d+$', c)]
                    for c in complex_cols:
                        device[c] = device[c].str.replace('i','j').apply(\
                            lambda x: np.complex(x))
                    self.device = device
                    n = len(self._eigen_names())
                    self.values = [self._eigen_value(i) for i in range(n)]
                    return self
                except:
                    pass

            # creates numpy arrays for hold the calculated values
            self.values = np.zeros(n, dtype=np.complex_)

            # create kickstart states
            # they consist of legendre polinomials modulated by a gaussian
            short_grid = np.linspace(-1, 1, self.N)
            g = gaussian(self.N, std=int(self.N/100))
            states = np.array([g * sp.legendre(i)(short_grid) \
                for i in range(n)],dtype=np.complex_)
            for i, state in enumerate(states):
                self.device['state_{}'.format(i)] = state

            for s in range(n):
                sn = 'state_{}'.format(s)
                for t in range(steps):
                    self.device[sn] = self.evolve_imag(self.device[sn], \
                        t0_au + t * self.dt_au)

                    # gram-shimdt
                    for m in range(s):
                        sm = 'state_{}'.format(m)
                        proj = simps(self.device[sn] * \
                            np.conjugate(self.device[sm]), self.device.x_au)
                        self.device[sn] -= proj * self.device[sm]

                    # normalize
                    self.device[sn] /= np.sqrt(simps(self.device[sn] * \
                        np.conjugate(self.device[sn]), self.device.x_au))

                self.values[s] = self._eigen_value(s)
            
            if save:
                if not os.path.exists(directory):
                    os.makedirs(directory)
                self.device.to_csv(full_filename)
        else:
            # this might be parallel
            for w in self._working_names():
                for t in range(steps):
                    self.device[w] = self.evolve_real(self.device[w], \
                        t0_au + t * self.dt_au)

        return self

    def normalize_device(self):
        """
        This function apply changes in the device structure or in external
        conditions to the main `device` object

        Returns
        -------
        self : HeteroStructure
            the current HeteroStructure object for further use in chain calls
        """
        try:
            device = self.device
        except:
            device = pd.DataFrame(dtype=np.complex_)
        
        # unique static inputs
        device['x_nm'] = np.copy(self.x_nm)
        device['v_ev'] = np.copy(self.v_ev)
        device['m_eff'] = np.copy(self.m_eff)

        # direct and reciprocal grids
        device['x_m'] = device.x_nm * 1.0e-9 # nm to m
        device['x_au'] = device.x_m / self.au_l # m to au
        self.dx_m = device.x_m.diff()[1] # dx
        self.dx_au = device.x_au.diff()[1] # dx
        device['k_au'] = fftfreq(self.N, d=self.dx_au)

        # static potential (ti = time independent)
        device['v_j'] = device.v_ev * self.ev # ev to j
        device['v_au_ti'] = device['v_au'] = device.v_j / self.au_e # j to au

        # check whether there is any bias to apply
        try:
            device['v_au_ti'] += self.bias_au
        except:
            pass

        # device is ready
        self.device = device

        # check whether there is any dynamic field to apply
        try:
            # v_au_td >> time dependent
            assert self.v_au_td and isinstance(self.v_au_td, LambdaType)
            self.v_au_full = lambda t: self.device.v_au_ti + self.v_au_td(t)
        except:
            self.v_au_full = lambda t: self.device.v_au_ti

        # imaginary time propagators
        exp_v2_i = lambda t: np.exp(- 0.5 * self.v_au_full(t) * self.dt_au)
        exp_t_i = np.exp(- 0.5 * (2 * np.pi * self.device.k_au) ** 2 * \
            self.dt_au / self.m_eff)
        self.evolve_imag = lambda psi, t: exp_v2_i(t) * ifft(exp_t_i * \
            fft(exp_v2_i(t) * psi))

        # normal propagators
        exp_v2 = lambda t: np.exp(- 0.5j * self.v_au_full(t) * self.dt_au)
        exp_t = np.exp(- 0.5j * (2 * np.pi * self.device.k_au) ** 2 * \
            self.dt_au / self.m_eff)
        self.evolve_real = lambda psi, t: exp_v2(t) * ifft(exp_t * \
            fft(exp_v2(t) * psi))

        return self

    # internals

    def _set_dt(self, dt=None):
        """
        this function might be used for setting the default time increment
        if *None* is given, then it will assume the value of `DEFAULT_DT`

        Parameters
        ----------
        dt : float
            the default time increment, in **seconds**
        """
        self.dt = dt or DEFAULT_DT
        self.dt_au = self.dt / self.au_t
    
    def _eigen_value(self, n, t=0.0):
        """ 
        **Only** for eigenfunctions of the current device

        Patameters
        ----------
        n : integer
            the eigenstate index
        t : float
            the time in atomic units, for use in time dependent potentials

        Returns
        -------
        eigenvalue : float
            the eigenvalue corresponding to the indicated eigenstate
        """
        sn = "state_{}".format(n)
        sn_st = "state_{}_conjugate".format(n)
        sn_d2 = "state_{}_2nd_derivative".format(n)
        
        # get only necessary columns
        device = self.device[['x_au', 'm_eff', sn]]

        # second derivative of psi_n
        device[sn_d2] = (device[sn].shift(1) - 2 * device[sn] + \
            device[sn].shift(-1)) / device['x_au'].diff()**2
        
        # remove those NA in the edges
        device = device.iloc[1:-1]

        # complex conjugate
        device[sn_st] = np.conjugate(device[sn])

        # <Psi|H|Psi>
        p_h_p = simps(device[sn_st] * (-0.5 * device[sn_d2] / device['m_eff'] \
            + self.v_au_full(t)[1:-1] * device[sn]), device['x_au'])
        # / <Psi|Psi> because I trust no one
        p_h_p /= simps(device[sn_st] * device[sn], device['x_au'])

        return p_h_p.real * self.au2ev # return value in eV

    def _eigen_names(self):
        """
        it returns the name of the columns where the eigenstates are stored in
        the main device

        Returns
        -------
        names : array_like
            the names of the columns in the device where eigenstates are stored
        """
        cols = self.device.columns
        return sorted([c for c in cols if re.match('^state_\d+$', c)])

    def _working_names(self):
        """
        it returns the name of the columns where the working waves are stored in
        the main device

        Returns
        -------
        names : array_like
            the names of the columns in the device where working waves are
            stored
        """
        cols = self.device.columns
        return sorted([c for c in cols if re.match('^working_\d+$', c)])

class BarriersWellSandwich(HeteroStructure):
    """
    """
    def __init__(self, b_l, d_l, w_l, b_x, d_x, w_x, N=None, bias=0.0, surround=2):
        """
        """
        super(BarriersWellSandwich,self).__init__()

        self.b_l_nm = b_l
        self.d_l_nm = d_l
        self.w_l_nm = w_l
        self.b_x_nm = b_x
        self.d_x_nm = d_x
        self.w_x_nm = w_x

        self.conduction_pct = 0.6
        self.valence_pct = 0.4

        self.core_length_nm = 2*b_l+2*d_l+w_l
        self.surround_times = surround # on each side

        self.system_length_nm = (2*self.surround_times + 1)*self.core_length_nm
        self.bulk_length_nm = (self.surround_times)*self.core_length_nm

        # this function return the number of points for a given length in nm
        self.pts = lambda l: int(l * float(self.N) / self.system_length_nm)

        self.x_nm = np.linspace(-self.system_length_nm/2, \
            self.system_length_nm/2, self.N)

        self.barrier = Database(Alloy.AlGaAs, b_x)
        self.span = Database(Alloy.AlGaAs, d_x)
        self.well = Database(Alloy.AlGaAs, w_x)

        span_cond_gap = self.conduction_pct * self.span.parameters('eg_0')
        barrier_cond_gap = self.conduction_pct * self.barrier.parameters('eg_0')
        well_cond_gap = self.conduction_pct * self.well.parameters('eg_0')
        span_meff = self.span.effective_masses('m_e')
        barrier_meff = self.barrier.effective_masses('m_e')
        well_meff = self.well.effective_masses('m_e')
        
        # bulk
        self.v_ev = self.pts(self.bulk_length_nm) * [span_cond_gap]
        self.m_eff = self.pts(self.bulk_length_nm) * [span_meff]
        self.points_before = len(self.v_ev)
        # first barrier
        self.v_ev += self.pts(b_l) * [barrier_cond_gap]
        self.m_eff += self.pts(b_l) * [barrier_meff]
        # first span
        self.v_ev += self.pts(d_l) * [span_cond_gap]
        self.m_eff += self.pts(d_l) * [span_meff]
        # well
        self.v_ev += self.pts(w_l) * [well_cond_gap]
        self.m_eff += self.pts(w_l) * [well_meff]
        # second span
        self.v_ev += self.pts(d_l) * [span_cond_gap]
        self.m_eff += self.pts(d_l) * [span_meff]
        # second barrier
        self.v_ev += self.pts(b_l) * [barrier_cond_gap]
        self.m_eff += self.pts(b_l) * [barrier_meff]
        self.points_after = len(self.v_ev)
        # span after second barrier
        self.v_ev += (self.N-len(self.v_ev)) * [span_cond_gap]
        self.m_eff += (self.N-len(self.m_eff)) * [span_meff]

        # shift zero to span potential
        self.v_ev = np.asarray(self.v_ev) - span_cond_gap

        # smooth the potential
        smooth_frac = int(float(self.N) / 500.0)
        self.v_ev = np.asarray([np.average(self.v_ev[max(0,i-smooth_frac):min(self.N-1,i+smooth_frac)]) for i in range(self.N)])

        # use numpy arrays
        self.m_eff = np.asarray(self.m_eff)

        self.normalize_device()

if __name__ == '__main__':
    system_properties = BarriersWellSandwich(5.0, 4.0, 5.0, 0.4, 0.2, 0.0, bias=0.0)
    
    #################### EIGENSTATES ###########################################
    info = system_properties.time_evolution(imaginary=True, n=1, steps=20000).get_system_state(0)
    eigenfunction, eigenvalue = info
    print(eigenvalue)