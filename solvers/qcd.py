#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains classes for simulating quantum heterostructures
based in  AlGaAs/GaAs devices
"""

import numpy as np

from .generic_potential import GenericPotential
from .band_structure_database import Alloy, Database

class QCD(GenericPotential):
    """[summary]
    
    Arguments:
        GenericPotential {[type]} -- [description]
    """

    def __init__(self, N=4096):
        super(QCD, self).__init__(N=N)
        
        L                = 1000.0
        z_ang            = np.linspace(0, L, N)
        xw               = 0.0
        xb               = 0.45
        wells            = [55,13,18,23,28,35,35,32,32,41,55]
        # barriers         = [65,60,43,36,34,56,40,56,35,33,65]
        barriers         = [65,60,43,36,34,56,40,56,35,33,0]
        structure_length = sum(wells)+sum(barriers)
        span_left        = (L - structure_length)/2
        span_right       = L - structure_length - span_left
        
        def x_shape(z):
            if z < span_left:
                return xb
            elif z > structure_length + span_left:
                return xb#xw
            else:
                dummy_accu = span_left
                for w,b in zip(wells, barriers):
                    dummy_accu += w
                    if z < dummy_accu:
                        return xw
                    dummy_accu += b
                    if z < dummy_accu:
                        return xb
            print(z)
        
        CBO, VBO = 0.7, 0.3
        g_algaas = lambda x: CBO*(1.519+1.447*x-0.15*x**2) # GAP
        m_algaas = lambda x: 0.067 # effective mass
        V        = np.vectorize(lambda z: g_algaas(x_shape(z)))(z_ang)
        V       -= g_algaas(xw)
        meff     = np.vectorize(lambda z: m_algaas(xw))(z_ang)

        # use numpy arrays
        self.v_ev  = np.array(V)
        self.m_eff = np.array(meff)
        self.z_nm  = np.array(z_ang) / 10.0

        self.normalize_device()