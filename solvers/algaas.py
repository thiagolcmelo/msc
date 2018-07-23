#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains classes for simulating quantum heterostructures
based in  AlGaAs/GaAs devices
"""

import numpy as np

from .generic_potential import GenericPotential
from .band_structure_database import Alloy, Database

class BarriersWellSandwichDegani(GenericPotential):
    """[summary]
    
    Arguments:
        GenericPotential {[type]} -- [description]
    """

    def __init__(self, N=4096):
        super(BarriersWellSandwichDegani, self).__init__(N=N)
        L         = 1000.0
        z_ang     = np.linspace(-L/2, L/2, N)
        g_algaas  = lambda x: 0.0 if x == 0.2 else (0.185897 if x == 0.4 else -0.185897)
        m_algaas  = lambda x: 0.067 # effective mass
        xd        = 0.2 # displacement
        xb        = 0.4 # barrier
        xw        = 0.0 # well
        wl        = 50.0 # Angstrom
        bl        = 50.0 # Angstrom
        dl        = 40.0 # Angstrom
        
        def x_shape(z):
            if np.abs(z) < wl/2:
                return xw
            elif np.abs(z) < wl/2+dl:
                return xd
            elif np.abs(z) < wl/2+dl+bl:
                return xb
            return xd

        V         = np.vectorize(lambda z: g_algaas(x_shape(z)))(z_ang)
        V        -= g_algaas(xd)
        meff      = np.vectorize(lambda z: m_algaas(xw))(z_ang)

        # use numpy arrays
        self.v_ev  = np.array(V)
        self.m_eff = np.array(meff)
        self.z_nm  = np.array(z_ang) / 10.0

        self.normalize_device()


class BarriersWellSandwich(GenericPotential):
    """
    This class provides a device which simulates a AlGaAs/Ga quantum 
    well surrounded by two barriers. The well and the barriers, as 
    well as the ground state potential are defined according to the
    concentration of Al.
    """

    def __init__(self, b_l, d_l, w_l, b_x, d_x, w_x, N=4096, \
        surround=2, offset=None, gap_distrib=(0.6,0.4)):
        """
        Parameters
        ----------
        b_l : float
            the barriers length in `nm`
        d_l : float
            the displacement length between barriers and the well 
            in `nm`
        w_l : float
            the well length in `nm`
        b_x : float
            the Al concentration in the barrier
        d_x : float
            the Al concentration in the displacement
        w_x : float
            the Al concentration in the well
        N : integer
            the number of points in the grid, the default id N = 4096
        surround : integer
            the core is composed of two barriers surrounding a well, 
            with a displacement between the barriers and the well in 
            both sides the `surround` is how many lengths equal to 
            the core's length will be surrounding the core on each side
        offset : string
            if not specified, there will be no offset, the allowed
            values are: 'span' and 'well'
        gap_distrib : tuple
            the gap distribution along conduction and valence bands
            (pct_conduction, pct_valence)
        """
        super(BarriersWellSandwich,self).__init__(N=N)

        self.b_l_nm = b_l
        self.d_l_nm = d_l
        self.w_l_nm = w_l
        self.b_x_nm = b_x
        self.d_x_nm = d_x
        self.w_x_nm = w_x

        self.conduction_pct, self.valence_pct = gap_distrib

        self.core_length_nm = 2 * b_l + 2 * d_l + w_l
        self.surround_times = surround # on each side

        self.system_length_nm = (2*self.surround_times + 1) * \
            self.core_length_nm
        self.bulk_length_nm = (self.surround_times)*self.core_length_nm

        # this function return the number of points 
        # for a given length in nm
        self.pts = \
            lambda l: int(l * float(self.N) / self.system_length_nm)

        self.z_nm = np.linspace(-self.system_length_nm/2, \
            self.system_length_nm/2, self.N)

        self.barrier = Database(Alloy.AlGaAs, b_x)
        self.span = Database(Alloy.AlGaAs, d_x)
        self.well = Database(Alloy.AlGaAs, w_x)

        span_cond_gap = self.conduction_pct * \
            self.span.parameters('eg_0')
        barrier_cond_gap = self.conduction_pct * \
            self.barrier.parameters('eg_0')
        well_cond_gap = self.conduction_pct * \
            self.well.parameters('eg_0')
        
        #span_meff = self.span.effective_masses('m_e')
        #barrier_meff = self.barrier.effective_masses('m_e')
        well_meff = self.well.effective_masses('m_e')
        span_meff = barrier_meff = well_meff
        
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

        # apply offset
        if offset == 'span':
            self.v_ev = np.asarray(self.v_ev) - span_cond_gap
        elif offset == 'well':
            self.v_ev = np.asarray(self.v_ev) - well_cond_gap

        # # smooth the potential
        # smooth_frac = int(float(self.N) / 500.0)
        # self.v_ev = np.asarray(\
        #     [np.average(self.v_ev[max(0,i-smooth_frac) : \
        #     min(self.N-1,i+smooth_frac)]) for i in range(self.N)])

        # use numpy arrays
        self.v_ev = np.array(self.v_ev)
        self.m_eff = np.array(self.m_eff)

        self.normalize_device()