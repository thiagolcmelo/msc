#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains classes for simulating generic devices
"""

import numpy as np

from heterostructures import HeteroStructure

class FiniteQuantumWell(HeteroStructure):
    """
    This class provides a device that simulates a simple quantum well, with a
    specific width and height
    """

    def __init__(self, wh, wl):
        """
        Parameters
        ----------
        wh : float
            is the well hight in eV
        wl : float
            is the well length in nm
        """
        super(FiniteQuantumWell,self).__init__(N=8192)

        self.wl = wl
        self.wh = wh

        self.surround_times = 75 # on each side
        self.system_length_nm = (2*self.surround_times + 1) * wl
        self.x_nm = np.linspace(-self.system_length_nm/2,\
            self.system_length_nm/2, self.N)

        self.pts = lambda l: int(l * float(self.N) / self.system_length_nm)
        self.bulk_length_nm = (self.surround_times)*self.wl

        # bulk
        self.v_ev = self.pts(self.bulk_length_nm) * [wh]
        self.m_eff = self.pts(self.bulk_length_nm) * [1.0]
        # well
        self.v_ev += self.pts(self.wl) * [0.0]
        self.m_eff += self.pts(self.wl) * [1.0]
        # bulk
        self.v_ev += (self.N-len(self.v_ev)) * [wh]
        self.m_eff += (self.N-len(self.m_eff)) * [1.0]

        # transforming to numpy arrays
        self.v_ev = np.array(self.v_ev)
        self.m_eff = np.array(self.m_eff)

        self.normalize_device()