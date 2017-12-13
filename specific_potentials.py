#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
this module contains some common potentials shapes, as well as properties
of heterostructures that fit such potentials
"""

class BarriersWellSandwich:
    """
    this class calculate properties for a potential such a quantum well
    surrounded by barriers
    the well and the barriers are "created" through different concentrations
    in the alloy used
    it does not use segragation, so in practice, it fits only AlGaAs
    requirements
    """
    def __init__(self, b_l, d_l, w_l, b_x, d_x, w_x):
        """
        Args:
        :b_l (float) is the barrier length in nanometers
        :d_l (float) is the span between the barriers and the well in nanometers
        :w_l (float) is the wells length in nanometers
        :b_x is the barrier 'x' concentration
        :d_x is the 'x' concentration of the span (and everything else)
        :w_x is the well 'x' concentration
        """
        self.b_l = b_l
        self.d_l = d_l
        self.w_l = w_l
        self.b_x = b_x
        self.d_x = d_x
        self.w_x = w_x

    def get_potential_shape(self):
        """
        return the potential shape in eV and nm in an dict like:
        {
            'potential': [...],
            'x': [...]
        }
        """
        return {
            'potential': self.v_ev,
            'x': []
        }
