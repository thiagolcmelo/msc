#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains classes for simulating quantum heterostructures based
in  AlGaAs/GaAs devices
"""

import numpy as np

from generic_potential import GenericPotential
from band_structure_database import Alloy, Database

class BarriersWellSandwich(GenericPotential):
    """
    This class provides a device which simulates a AlGaAs/Ga quantum well 
    surrounded by two barriers. The well and the barriers, as well as the
    ground state potential are defined according to the concentration of Al.
    """

    def __init__(self, b_l, d_l, w_l, b_x, d_x, w_x, N=8192, surround=2):
        """
        Parameters
        ----------
        b_l : float
            the barriers length in `nm`
        d_l : float
            the displacement length between barriers and the well in `nm`
        w_l : float
            the well length in `nm`
        b_x : float
            the Al concentration in the barrier
        d_x : float
            the Al concentration in the displacement
        w_x : float
            the Al concentration in the well
        N : integer
            the number of points in the grid, the default id N = 8192
        surround : integer
            the core is composed of two barriers surrounding a well, with a
            displacement between the barriers and the well in both sides
            the `surround` is how many lengths equal to the core's length will
            be surrounding the core on each side
        """
        super(BarriersWellSandwich,self).__init__(N=N)

        self.b_l_nm = b_l
        self.d_l_nm = d_l
        self.w_l_nm = w_l
        self.b_x_nm = b_x
        self.d_x_nm = d_x
        self.w_x_nm = w_x

        self.conduction_pct = 0.6
        self.valence_pct = 0.4

        self.core_length_nm = 2 * b_l + 2 * d_l + w_l
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
        self.v_ev = np.asarray([np.average(self.v_ev[max(0,i-smooth_frac) : \
            min(self.N-1,i+smooth_frac)]) for i in range(self.N)])

        # use numpy arrays
        self.m_eff = np.asarray(self.m_eff)

        self.normalize_device()