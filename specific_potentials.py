#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
this module contains some common potentials shapes, as well as properties
of heterostructures that fit such potentials
"""
import math
import numpy as np
from band_structure_database import Alloy, Database

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
        self.b_l_nm = b_l
        self.d_l_nm = d_l
        self.w_l_nm = w_l
        self.b_x_nm = b_x
        self.d_x_nm = d_x
        self.w_x_nm = w_x

        self.conduction_pct = 0.6
        self.valence_pct = 0.4

        self.core_length_nm = 2*b_l+2*d_l+w_l
        self.surround_times = 6 # an even number will be better
        self.system_length_nm = (self.surround_times + 1)*self.core_length_nm
        self.bulk_length_nm = (self.surround_times/2)*self.core_length_nm

        # weird rule
        # 20 A -> 128 pts
        # 20 A * sqrt(2) -> 2*128=256 pts
        self.N = 2 ** int(math.log2(128 * (self.system_length_nm / 2.0) ** 2))
        # this function return the number of points for a given length
        self.pts = lambda l: int(l * float(self.N) / self.system_length_nm)

        self.x_nm = np.linspace(-self.system_length_nm/2, \
            self.system_length_nm/2, self.N)

        self.barrier = Database(Alloy.AlGaAs, b_x)
        self.span = Database(Alloy.AlGaAs, d_x)
        self.well = Database(Alloy.AlGaAs, w_x)

        # build the potential
        span_cond_gap = self.conduction_pct * self.span.parameters('eg_0')
        barrier_cond_gap = self.conduction_pct * self.barrier.parameters('eg_0')
        well_cond_gap = self.conduction_pct * self.well.parameters('eg_0')

        # bulk
        self.v_ev = self.pts(self.bulk_length_nm) * [span_cond_gap]
        # first barrier
        self.v_ev += self.pts(b_l) * [barrier_cond_gap]
        # first span
        self.v_ev += self.pts(d_l) * [span_cond_gap]
        # well
        self.v_ev += self.pts(w_l) * [well_cond_gap]
        # second span
        self.v_ev += self.pts(d_l) * [span_cond_gap]
        # second barrier
        self.v_ev += self.pts(b_l) * [barrier_cond_gap]
        # span after second barrier
        self.v_ev += (self.N-len(self.v_ev)) * [span_cond_gap]

        # shift zero to span potential
        self.v_ev = np.asarray(self.v_ev) - span_cond_gap

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
            'x': self.x_nm
        }

    def generate_eigenfunctions(self, n=3):
        """
        this will generate :n first eigenfunctions (and eigenvalues)

        Args:
        :n (int) number of eigenfunctions to be calculated
        """




if __name__ == '__main__':
    system_properties = BarriersWellSandwich(5.0, 9.0, 8.2, 0.4, 0.3, 0.0)
    potential_shape = system_properties.get_potential_shape()
    import matplotlib.pyplot as plt
    plt.plot(potential_shape['x'], potential_shape['potential'])
    plt.show()
