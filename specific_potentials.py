#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
this module contains some common potentials shapes, as well as properties
of heterostructures that fit such potentials
"""
import numpy as np
from scipy.integrate import simps
import scipy.constants as cte
import scipy.special as sp
from scipy.signal import gaussian
from scipy.fftpack import fft, ifft, fftfreq

from band_structure_database import Alloy, Database

class GenericPotential(object):
    """
    
    """

    def __init__(self, N=None):
        # default grid size
        self.N = N or 8192

        # atomic unities
        self.au_l = cte.value('atomic unit of length')
        self.au_t = cte.value('atomic unit of time')
        self.au_e = cte.value('atomic unit of energy')
        self.hbar_au = 1.0
        self.me_au = 1.0

        # other useful constants
        self.ev = cte.value('electron volt')
        self.c = cte.value('speed of light in vacuum') # m/s
        self.me = cte.value('electron mass')
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

    def get_potential_shape(self):
        """
        return the potential shape in eV and nm in an dict like:
        {
            'potential': [...],
            'x': [...],
            'm_eff': [...],
        }
        """
        return {
            'potential': self.v_ev,
            'x': self.x_nm,
            'm_eff': self.m_eff
        }

    def generate_eigenfunctions(self, n=3, steps=2000, dt=None):
        """
        ### Generating eigenvalues and eigenfunctions
        this will generate `n` first eigenfunctions (and eigenvalues)

        Args:

        `n` (int) number of eigenfunctions to be calculated

        `steps` (int) is the number of time evolutions per state
        """

        # translate properties to atomic unities
        self.x_m = self.x_nm * 1.0e-9 # nm to m
        self.x_au = self.x_m / self.au_l # m to au
        self.dx_au = self.x_au[1]-self.x_au[0] # dx
        
        self.v_j = self.v_ev * self.ev # ev to j
        self.v_au = self.v_j / self.au_e # j to au

        # NOT SURE YET
        self.dt_s = dt or 1.0e-16
        self.dt_au = self.dt_s / self.au_t # s to au

        # k grid
        self.k_au = fftfreq(self.N, d=self.dx_au)

        # creates numpy arrays for hold the calculated values
        states = np.zeros((n, self.N), dtype=np.complex_)
        values = np.zeros(n, dtype=np.complex_)

        # create kickstart states
        # they consist of legendre polinomials modulated by a gaussian
        short_grid = np.linspace(-1, 1, self.N)
        g = gaussian(self.N, std=int(self.N/100))
        states = np.asarray([g * sp.legendre(i)(short_grid) for i in range(n)],\
             dtype=np.complex_)

        # evolve in imaginary time
        exp_v2 = np.exp(- 0.5 * self.v_au * self.dt_au)
        exp_t = np.exp(- 0.5 * (2 * np.pi * self.k_au) ** 2 * self.dt_au / self.m_eff)
        
        def eigen_value(psi):
            """ calculate eigenvalue for a eigenstate given the current
            calculations potential, masses, and so on"""
            second_derivative = np.asarray(psi[0:-2]-2*psi[1:-1]+psi[2:]) \
                /self.dx_au**2
            psi = psi[1:-1]
            psi_st = np.conjugate(psi)
            
            me = np.asarray(self.m_eff[1:-1])
            
            h_p_h = simps(psi_st * (-0.5 * second_derivative / me + \
                self.v_au[1:-1] * psi), self.x_au[1:-1])
            return h_p_h.real * self.au2ev
        
        evolve_once = lambda psi: exp_v2 * ifft(exp_t * fft(exp_v2 * psi))

        for s in range(n):

            for t in range(steps):
                # evolve once
                states[s] = evolve_once(states[s])

                # gram-shimdt
                for m in range(s):
                    proj = simps(states[s] * np.conjugate(states[m]), self.x_au)
                    states[s] -= proj * states[m]

                # normalize
                states[s] /= np.sqrt(simps(states[s] * np.conjugate(states[s]), self.x_au))

                if t % 10 == 0:
                    print('t = %d, E_%d: %.4f meV' % (t, s, 1000*eigen_value(states[s])))
                
        for s in range(n):
            # calculate eigenvalue
            values[s] = eigen_value(states[s])

        return {
            'eigenstates': states,
            'eigenvalues': values
        }

class FiniteQuantumWell(GenericPotential):
    """
    """

    def __init__(self, wh, wl):
        """

        `wh` is the well hight in eV
        `wl` is the well length in nm
        """
        super(FiniteQuantumWell,self).__init__(N=8192)

        self.wl = wl
        self.wh = wh

        self.surround_times = 150 # half on each side
        self.system_length_nm = (self.surround_times + 1) * wl
        self.x_nm = np.linspace(-self.system_length_nm/2,\
            self.system_length_nm/2, self.N)

        self.pts = lambda l: int(l * float(self.N) / self.system_length_nm)
        self.bulk_length_nm = (self.surround_times/2)*self.wl

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

class BarriersWellSandwich(GenericPotential):
    """
    this class calculate properties for a potential such a quantum well
    surrounded by barriers
    the well and the barriers are "created" through different concentrations
    in the alloy used
    it does not use segragation, so in practice, it fits only AlGaAs
    requirements
    """
    def __init__(self, b_l, d_l, w_l, b_x, d_x, w_x, N=None):
        """
        Args:
        :b_l (float) is the barrier length in nanometers
        :d_l (float) is the span between the barriers and the well in nanometers
        :w_l (float) is the wells length in nanometers
        :b_x is the barrier 'x' concentration
        :d_x is the 'x' concentration of the span (and everything else)
        :w_x is the well 'x' concentration
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
        self.surround_times = 2 # use a even number

        self.system_length_nm = (self.surround_times + 1)*self.core_length_nm
        self.bulk_length_nm = (self.surround_times/2)*self.core_length_nm

        # weird rule
        # 20 A -> 128 pts
        # 20 A * sqrt(2) -> 2*128=256 pts
        #self.N = 2 ** int(np.log2(128 * (self.system_length_nm / 2.0) ** 2))
        # 1024, 100,000 [-0.15769882+0.j  0.05241221+0.j]
        #self.N = 8192

        # this function return the number of points for a given length in nm
        self.pts = lambda l: int(l * float(self.N) / self.system_length_nm)

        self.x_nm = np.linspace(-self.system_length_nm/2, \
            self.system_length_nm/2, self.N)

        self.barrier = Database(Alloy.AlGaAs, b_x)
        self.span = Database(Alloy.AlGaAs, d_x)
        self.well = Database(Alloy.AlGaAs, w_x)

        # build the potential
        #vd = lambda x: 0.6*(1.425+1.155*x+0.37*x**2)
        #vd = lambda x: 0.6*(1.155*x+0.37*x**2)
        #span_cond_gap = vd(d_x)
        #barrier_cond_gap = vd(b_x)
        #well_cond_gap = vd(w_x)
        #span_meff = 0.067
        #barrier_meff = 0.067
        #well_meff = 0.067

        span_cond_gap = self.conduction_pct * self.span.parameters('eg_0')
        barrier_cond_gap = self.conduction_pct * self.barrier.parameters('eg_0')
        well_cond_gap = self.conduction_pct * self.well.parameters('eg_0')
        span_meff = self.span.effective_masses('m_e')
        barrier_meff = self.barrier.effective_masses('m_e')
        well_meff = self.well.effective_masses('m_e')
        
        
        # bulk
        self.v_ev = self.pts(self.bulk_length_nm) * [span_cond_gap]
        self.m_eff = self.pts(self.bulk_length_nm) * [span_meff]
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
        # span after second barrier
        self.v_ev += (self.N-len(self.v_ev)) * [span_cond_gap]
        self.m_eff += (self.N-len(self.m_eff)) * [span_meff]

        # shift zero to span potential
        self.v_ev = np.asarray(self.v_ev) - span_cond_gap

        # smooth the potential
        #smooth_frac = int(float(self.N) / 500.0)
        #self.v_ev = np.asarray([np.average(self.v_ev[max(0,i-smooth_frac):min(self.N-1,i+smooth_frac)]) for i in range(self.N)])

        # use numpy arrays
        self.m_eff = np.asarray(self.m_eff)

class MultiQuantumWell(GenericPotential):
    """

    """

    def __init__(self, b_x=0.382, w_x=0.0, w_n=9, total_length=600.0):
        """
        The default values are those used by the article

        """
        super(MultiQuantumWell,self).__init__(N=8192)

        self.barrier = Database(Alloy.AlGaAs, b_x)
        self.well = Database(Alloy.AlGaAs, w_x)
        
        self.conduction_pct = 0.6
        self.valence_pct = 0.4

        barrier_cond_gap = 1.28#self.conduction_pct * self.barrier.parameters('eg_0')
        well_cond_gap = 0.0#self.conduction_pct * self.well.parameters('eg_0')
        barrier_meff = self.barrier.effective_masses('m_e')
        well_meff = self.well.effective_masses('m_e')

        self.w_n = w_n
        self.system_length_nm = total_length
        self.w_l = self.system_length_nm / float(2 * w_n + 1)
        self.pts = lambda l: int(l * float(self.N) / self.system_length_nm)

        self.v_ev = []
        self.m_eff = []
        for i in range(self.w_n):
            self.v_ev += self.pts(self.w_l) * [barrier_cond_gap]
            self.m_eff += self.pts(self.w_l) * [barrier_meff]
            self.v_ev += self.pts(self.w_l) * [well_cond_gap]
            self.m_eff += self.pts(self.w_l) * [well_meff]

        self.v_ev += (self.N-len(self.v_ev)) * [barrier_cond_gap]
        self.m_eff += (self.N-len(self.m_eff)) * [barrier_meff]

        self.v_ev = np.array(self.v_ev)
        self.m_eff = np.array(self.m_eff)

        self.x_nm = np.linspace(-self.system_length_nm/2, \
            self.system_length_nm/2, self.N)


if __name__ == '__main__':
    #import numpy as np
    #import scipy.constants as cte
    #hbar=cte.value('Planck constant over 2 pi')
    #c=cte.value('speed of light in vacuum')
    #ev = cte.value('electron volt')
    #w=2*np.pi*c/(0.0000081)
    #f = np.vectorize(lambda n: hbar * w * (0.5+n) / ev)
    #print(f(range(3)))

    #system_properties = BarriersWellSandwich(5.0, 9.0, 8.2, 0.4, 0.3, 0.0)
    #system_properties = GenericPotential()
    system_properties = MultiQuantumWell(w_n=2, total_length=150.0)
    #system_properties = FiniteQuantumWell(wh=25.0, wl=0.5)
    potential_shape = system_properties.get_potential_shape()
    
    import matplotlib.pyplot as plt
    plt.plot(potential_shape['x'], potential_shape['potential'])
    #plt.plot(potential_shape['x'], potential_shape['m_eff'])
    result = system_properties.generate_eigenfunctions(20, steps=10000)
    for i, p in enumerate(result['eigenstates']):
        plt.plot(potential_shape['x'], (p*np.conjugate(p)).real+result['eigenvalues'][i])
    print(result['eigenvalues'])
    plt.show()

    #for dt in [1e-16, 5e-17, 1e-17, 5e-18]:
    #    for N in 2**np.asarray(range(12,20)):
    #        for steps in [2000, 5000, 10000, 20000]:
    #            system_properties = BarriersWellSandwich(5.0, 9.0, 8.2, 0.4, 0.3, 0.0, N=N)
    #            result = system_properties.generate_eigenfunctions(2, steps=steps, dt=dt)
    #            eigenvalues = result['eigenvalues']
    #            print('N=%d, steps=%d, dt=%.1e s, e0=%.6f meV, e1=%.6f meV' % (N, steps, dt, 1000*eigenvalues[0], 1000*eigenvalues[1]))

                