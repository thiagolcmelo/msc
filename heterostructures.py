#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains classes for simalating quantum heterostructures
"""

import numpy as np
import pandas as pd

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

import os, time
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


    # operations


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
        device.diff()
        # unique static inputs
        device['x_nm'] = np.copy(self.x_nm)
        device['v_ev'] = np.copy(self.v_ev)

        # direct and reciprocal grid
        device['x_m'] = device.x_nm * 1.0e-9 # nm to m
        device['x_au'] = device.x_m / self.au_l # m to au
        self.dx_m = device.x_m[1]-device.x_m[0] # dx
        self.dx_au = device.x_au[1]-device.x_au[0] # dx
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
    
    def _eigen_value(self, n):
        """ 
        **Only** for eigenfunctions of the object's potential

        Patameters
        ----------
        n : integer
            the eigenstate index

        Returns
        -------
        eigenvalue : float
            the eigenvalue corresponding to the indicated eigenstate
        """
        psi = np.copy(self.states[n])
        second_derivative = np.asarray(psi[0:-2]-2*psi[1:-1]+psi[2:]) \
            /self.dx_au**2
        psi = psi[1:-1]
        psi_st = np.conjugate(psi)
        me = np.asarray(self.m_eff[1:-1])
        h_p_h = simps(psi_st * (-0.5 * second_derivative / me + \
            self.v_au[1:-1] * psi), self.x_au[1:-1])
        h_p_h /= simps(psi_st * psi, self.x_au[1:-1])
        return h_p_h.real * self.au2ev


    # (df['f'].shift(1)-2*df['f']+df['f'].shift(-1))/(df['x'].diff()**2)