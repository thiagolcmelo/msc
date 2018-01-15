#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
this module contains some common potentials shapes, as well as properties
of heterostructures that fit such potentials
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.integrate import simps
import scipy.constants as cte
import scipy.special as sp
from scipy.signal import gaussian
from scipy.fftpack import fft, ifft, fftfreq
from datetime import datetime

from multiprocessing import Pool, TimeoutError
import time
import os

from band_structure_database import Alloy, Database

class GenericPotential(object):
    """
    
    """

    def __init__(self, N=None):
        # default grid size
        self.N = N or 8192
        self.hN = int(self.N/2)
        self.points_before = int(self.N/2)
        self.points_after = int(self.N/2)

        # atomic unities
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

        # very useful
        self.absolute2 = np.vectorize(lambda x: np.absolute(x)**2)
        self.self_inner_prod = lambda f,x: simps(self.absolute2(f), x)

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

    def eigen_value(self, psi):
        """ calculate eigenvalue for a eigenstate given the current
        calculations potential, masses, and so on"""
        second_derivative = np.asarray(psi[0:-2]-2*psi[1:-1]+psi[2:]) \
            /self.dx_au**2
        psi = psi[1:-1]
        psi_st = np.conjugate(psi)
        
        me = np.asarray(self.m_eff[1:-1])
        
        h_p_h = simps(psi_st * (-0.5 * second_derivative / me + \
            self.v_au[1:-1] * psi), self.x_au[1:-1])
        h_p_h /= simps(psi_st * psi, self.x_au[1:-1])
        return h_p_h.real * self.au2ev

    def wave_energy(self, psi):
        energy = 0.0
        for value, state in zip(self.values, self.states):
            an = simps(state.conjugate()*psi, self.x_au)/simps(state.conjugate()*state, self.x_au)
            energy += (an.conjugate()*an)*value
        return energy

    def generate_eigenfunctions(self, n=3, steps=2000, dt=None, verbose=False):
        """
        """

        # translate properties to atomic unities
        self.x_m = self.x_nm * 1.0e-9 # nm to m
        self.x_au = self.x_m / self.au_l # m to au
        self.dx_au = self.x_au[1]-self.x_au[0] # dx
        
        # k grid
        self.k_au = fftfreq(self.N, d=self.dx_au)

        # et >>> j >>> au
        self.v_j = self.v_ev * self.ev # ev to j
        self.v_au = self.v_j / self.au_e # j to au

        # NOT SURE YET
        self.dt_s = dt or 1.0e-18
        self.dt_au = self.dt_s / self.au_t # s to au

        # creates numpy arrays for hold the calculated values
        self.states = np.zeros((n, self.N), dtype=np.complex_)
        self.values = np.zeros(n, dtype=np.complex_)

        # create kickstart states
        # they consist of legendre polinomials modulated by a gaussian
        short_grid = np.linspace(-1, 1, self.N)
        g = gaussian(self.N, std=int(self.N/100))
        self.states = np.asarray([g * sp.legendre(i)(short_grid) for i in range(n)],\
             dtype=np.complex_)

        # evolve in imaginary time
        exp_v2 = np.exp(- 0.5 * self.v_au * self.dt_au)
        exp_t = np.exp(- 0.5 * (2 * np.pi * self.k_au) ** 2 * self.dt_au / self.m_eff)
        
        evolve_once = lambda psi: exp_v2 * ifft(exp_t * fft(exp_v2 * psi))

        for s in range(n):
            for t in range(steps):
                # evolve once
                self.states[s] = evolve_once(self.states[s])

                # gram-shimdt
                for m in range(s):
                    proj = simps(self.states[s] * np.conjugate(self.states[m]), self.x_au)
                    self.states[s] -= proj * self.states[m]

                # normalize
                self.states[s] /= np.sqrt(simps(self.states[s] * np.conjugate(self.states[s]), self.x_au))
                self.states[s] /= np.sqrt(simps(np.absolute(self.states[s]) ** 2, self.x_au))

                if verbose and t % 100 == 0:
                    ev = self.eigen_value(self.states[s])
                    print('t = %d, E_%d: %.10f meV' % (t, s, 1000*ev))
            
            self.values[s] = self.eigen_value(self.states[s])
            if verbose:
                print("E_%d = %.10f eV" % (s, self.values[s]))

        return {
            'eigenstates': self.states,
            'eigenvalues': self.values
        }

    def ajust_unities(self):
        # translate properties to atomic unities
        self.x_m = self.x_nm * 1.0e-9 # nm to m
        self.x_au = self.x_m / self.au_l # m to au
        self.dx_au = self.x_au[1]-self.x_au[0] # dx
        
        self.v_j = self.v_ev * self.ev # ev to j
        self.v_au = self.v_j / self.au_e # j to au

        # k grid
        self.k_au = fftfreq(self.N, d=self.dx_au)

        # NOT SURE YET
        self.dt_s = 1.0e-18
        self.dt_au = self.dt_s / self.au_t # s to au

    def evolve_pulse(self, steps=2000, dt=None, display=True):
        """
        """
        # NOT SURE YET
        self.dt_s = dt or 1.0e-18
        self.dt_au = self.dt_s / self.au_t # s to au
        
        exp_v2 = np.exp(- 0.5j * (self.v_au + self.v_au_abs) * self.dt_au)
        exp_t = np.exp(- 0.5j * \
            (2 * np.pi * self.k_au) ** 2 * self.dt_au / self.m_eff)
        
        evolve_once = lambda psi: exp_v2 * ifft(exp_t * fft(exp_v2 * psi))
        self.max_trans = 0.0
        total = simps(self.pulse*self.pulse.conjugate(), self.x_au)
        #real_energy = self.wave_energy(self.pulse)
        pb = self.points_before
        pa = self.points_after

        if display: 
            fig, ax = plt.subplots()
            
            ax.plot(self.x_nm, self.v_ev)
            ax.plot(self.x_nm, self.v_au_abs.imag * self.au2ev)

            pulse_height = np.ptp(self.v_ev) / 5
            start_norm = np.ptp(self.pulse*self.pulse.conjugate())
            fake_e0 = self.pulse_E_ev

            visual_pulse = lambda p: pulse_height * p*p.conjugate() / start_norm + 0.05#fake_e0
            #visual_pulse = lambda p: p*p.conjugate()
            line, = ax.plot(self.x_nm, visual_pulse(self.pulse))
            #a_text = ax.text(0.03, 0.90, '', transform=ax.transAxes)
            time_text = ax.text(0.03, 0.95, '', transform=ax.transAxes)

            ax.grid(True)
            ax.set_xlabel('$x (nm)$')
            ax.set_ylabel('$E (eV)$')
            
            max_trans_text = ax.text(0.02, 0.05, '', transform=ax.transAxes)
            trans_text = ax.text(0.02, 0.10, '', transform=ax.transAxes)
            refle_text = ax.text(0.02, 0.15, '', transform=ax.transAxes)
            bound_text = ax.text(0.02, 0.20, '', transform=ax.transAxes)
            energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
            energy_text.set_text("energy: %.3f eV" % (self.pulse_E_ev))

            def animate(i):
                for _ in range(100):
                    self.pulse = evolve_once(self.pulse)
                line.set_ydata(visual_pulse(self.pulse))
                #energy_text.set_text("A = %.6f" % (simps(self.pulse * np.conjugate(self.pulse), self.x_au)))
                time_text.set_text("t = %.3e sec" % (self.dt_s * float(i)))
                
                self.refle = (simps(self.pulse[:pb]*self.pulse.conjugate()[:pb], self.x_au[:pb])/total).real
                self.trans = (simps(self.pulse[pa:]*self.pulse.conjugate()[pa:], self.x_au[pa:])/total).real
                self.bound = (simps(self.pulse[pb:pa]*self.pulse.conjugate()[pb:pa], self.x_au[pb:pa])/total).real
                self.max_trans = max(self.max_trans, self.trans)

                max_trans_text.set_text("max trans: %.6f %%" % (100*self.max_trans))
                trans_text.set_text("trans: %.6f %%" % (100*self.trans))
                refle_text.set_text("refle: %.6f %%" % (100*self.refle))
                bound_text.set_text("bound: %.6f %%" % (100*self.bound))

                return line,energy_text, time_text, refle_text, trans_text, max_trans_text, bound_text

            def init():
                line.set_ydata(np.ma.array(self.x_nm, mask=True))
                return line,
            ani = animation.FuncAnimation(fig, animate, np.arange(1, 200), init_func=init, interval=25, blit=True)
            plt.show()

            # self.pulse /= np.sqrt(simps(self.pulse * np.conjugate(self.pulse), self.x_au))

        else:
            for t in range(steps):
                self.pulse = evolve_once(self.pulse)
                #self.refle = (simps(self.pulse[:pb]*self.pulse.conjugate()[:pb], self.x_au[:pb])/total).real
                self.trans = (simps(self.pulse[pa:]*self.pulse.conjugate()[pa:], self.x_au[pa:])/total).real
                self.max_trans = max(self.max_trans, self.trans)

                #if t % 1000 == 0:
                    #print("refle: %.6f, bound: %.6f, trans: %.6f, max trans: %.6f" % (100*self.refle, 100*self.bound, 100*self.trans, 100*self.max_trans))
                
            #print("refle: %.6f, bound: %.6f, trans: %.6f, max trans: %.6f" % (100*self.refle, 100*self.bound, 100*self.trans, 100*self.max_trans))

        self.bound = (simps(self.pulse[pb:pa]*self.pulse.conjugate()[pb:pa], self.x_au[pb:pa])/total).real
        return (self.pulse, self.max_trans, self.bound)

    def gaussian_pulse(self, delta_x, x0, E, direction='L2R'):
        assert self.x_nm[0] < x0 < self.x_nm[-1]
        assert delta_x < np.ptp(self.x_nm)
        assert direction in ['L2R', 'R2L']

        self.pulse_E_ev = E
        self.pulse_E_au = E * self.ev / self.au_e
        self.pulse_x0_nm = x0 * 1e-9
        self.pulse_x0_au = x0 * 1e-9 / self.au_l

        self.pulse_x0_index = np.searchsorted(self.x_au, self.pulse_x0_au)
        
        self.pulse_x0_meff = self.m_eff[self.pulse_x0_index]
        
        self.pulse_k0_au = np.sqrt(2.0 * self.pulse_x0_meff * self.pulse_E_au)
        self.pulse_k0_au *= 1.0 if direction == 'L2R' else -1.0

        delta_e_au = 0.001 / self.au2ev
        delta_k_au = np.sqrt(2.0 * delta_e_au * self.pulse_x0_meff)
        delta_x_au = 0.5 / delta_k_au # uncertainty principle
        
        #self.pulse_delta_x_nm = delta_x * 1e-9
        #self.pulse_delta_x_au = delta_x * 1e-9 / self.au_l
        self.pulse_delta_x_au = delta_x_au
        
        self.wave_amp = (2.0 * np.pi * self.pulse_delta_x_au ** 2) ** (0.25)
        self.pulse_func = lambda x: self.wave_amp * \
            np.exp((1j) * self.pulse_k0_au * x - \
            (x - self.pulse_x0_au) ** 2 / (4.0 * self.pulse_delta_x_au ** 2))
        self.pulse_func = np.vectorize(self.pulse_func)
        self.pulse = self.pulse_func(self.x_au)

        ###################################
        a5 = 1.12 * self.pulse_E_au
        #abs_pts = int(self.N / 15)
        v_abs = np.vectorize(lambda y: -1j*a5*(13.22*np.exp(-2.0/y)))

        qN = int(self.hN/8)
        self.v_au_abs = v_abs(np.linspace(1e-9, 1, qN))
        self.v_au_abs = np.append(np.append(self.v_au_abs[::-1], np.zeros(self.hN + 6*qN)), self.v_au_abs)
        #self.v_au_abs = v_abs(np.linspace(1e-9, 1, self.hN))
        #self.v_au_abs = np.append(self.v_au_abs[::-1], self.v_au_abs)
        #self.v_au_abs = np.zeros(self.N)
        return self.pulse

    def photocurrent(self, energy, T=1.0e-12, Fdyn=5.0, dt=None):
        self.energy_ex_ev = energy
        self.energy_ex_au = energy /  self.au2ev
        self.T = T
        self.T_au = T / self.au_t

        self.dt_s = dt or 1.0e-17
        self.dt_au = self.dt_s / self.au_t # s to au

        # KV/cm
        self.Fdyn_v_cm = Fdyn * 1000.0
        self.Fdyn_v_m = 100.0 * self.Fdyn_v_cm
        self.Fdyn_j_m = self.Fdyn_v_m * self.q
        self.Fdyn_j = np.vectorize(lambda z: (self.x_m[0] - z) * self.Fdyn_j_m)(self.x_m)
        self.Fdyn_ev = self.Fdyn_j / self.ev
        self.Fdyn_au = self.Fdyn_ev / self.au2ev
        
        self.omega_au = self.energy_ex_au / self.hbar_au
        exp_t = np.exp(- 0.5j * (2 * np.pi * self.k_au) ** 2 * self.dt_au / self.m_eff)

        self.PSI = self.states[0]

        ###################################
        a5 = 1.12 * self.values[0] / self.au2ev
        #abs_pts = int(self.N / 15)
        v_abs = np.vectorize(lambda y: -1j*a5*(13.22*np.exp(-2.0/y)))
        qN = int(self.hN/8)
        self.v_au_abs = v_abs(np.linspace(1e-9, 1, qN))
        self.v_au_abs = np.append(np.append(self.v_au_abs[::-1], np.zeros(self.hN + 6*qN)), self.v_au_abs)

        self.j_t = []
        self.t_grid_au = np.linspace(0.0, self.T_au, int(self.T_au / self.dt_au))
        
        pb = self.points_before - 100
        pa = self.points_after + 100

        for i, t_au in enumerate(self.t_grid_au):
            #exp_v2 = np.exp(- 0.5j * (self.v_au + self.v_au_abs + self.Fdyn_au * np.sin(self.omega_au*t_au)) * self.dt_au)
            exp_v2 = np.exp(- 0.5j * (self.v_au + self.Fdyn_au * np.sin(self.omega_au*t_au)) * self.dt_au)
            evolve_once = lambda psi: exp_v2 * ifft(exp_t * fft(exp_v2 * psi))
            self.PSI = evolve_once(self.PSI)
            
            j_l = ((-0.5j/(self.m_eff[pb])) * (self.PSI[pb].conjugate() * (self.PSI[pb+1]-self.PSI[pb-1]) - self.PSI[pb] * (self.PSI[pb+1].conjugate()-self.PSI[pb-1].conjugate())) / (2*self.dx_au)).real
            j_r = ((-0.5j/(self.m_eff[pa])) * (self.PSI[pa].conjugate() * (self.PSI[pa+1]-self.PSI[pa-1]) - self.PSI[pa] * (self.PSI[pa+1].conjugate()-self.PSI[pa-1].conjugate())) / (2*self.dx_au)).real

            self.j_t.append(j_r-j_l)
            #if i % 1000 == 0:
            #    print(i)
        
        #plt.plot(self.t_grid_au, self.j_t)
        #plt.show()
        #return self.q * simps(self.j_t, self.t_grid_au) / self.T_au
        return simps(self.j_t, self.t_grid_au) / self.T_au

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

        self.ajust_unities()

class BarriersWellSandwich(GenericPotential):
    """
    this class calculate properties for a potential such a quantum well
    surrounded by barriers
    the well and the barriers are "created" through different concentrations
    in the alloy used
    it does not use segragation, so in practice, it fits only AlGaAs
    requirements
    """
    def __init__(self, b_l, d_l, w_l, b_x, d_x, w_x, N=None, Fst=0.0, surround=2):
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
        self.surround_times = surround # on each side

        self.system_length_nm = (2*self.surround_times + 1)*self.core_length_nm
        self.bulk_length_nm = (self.surround_times)*self.core_length_nm

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

        self.ajust_unities()

        # apply the bias
        if Fst != 0.0:
            # KV/cm
            self.Fst_v_cm = Fst * 1000.0
            self.Fst_v_m = 100.0 * self.Fst_v_cm
            self.Fst_j_m = self.Fst_v_m * self.q

            #def Fst_shape(z):
            #    i = np.searchsorted(self.x_m, z)
            #    if i < self.points_before:
            #        return 0.0
            #    elif self.points_before < i < self.points_after:
            #        return (self.x_m[self.points_before] - z) * self.Fst_j_m
            #    else:
            #        return -self.x_m[self.points_after] * self.Fst_j_m

            self.Fst_j = np.vectorize(lambda z: (self.x_m[0] - z) * self.Fst_j_m)(self.x_m)
            
            self.Fst_ev = self.Fst_j / self.ev
            self.Fst_au = self.Fst_ev / self.au2ev
            self.v_ev += self.Fst_ev

            self.ajust_unities()

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

class DoubleBarrier(GenericPotential):
    def __init__(self, b_l, w_l, b_h, w_h, N=None, Fst=0.0, surround=1):
        super(DoubleBarrier,self).__init__()

        self.b_l_nm = b_l
        self.w_l_nm = w_l
        self.b_h_ev = b_h
        self.w_h_ev = w_h

        self.core_length_nm = 2*b_l+w_l
        self.surround_times = surround # on each side

        self.system_length_nm = (2*self.surround_times + 1)*self.core_length_nm
        self.bulk_length_nm = (self.surround_times)*self.core_length_nm

        # this function return the number of points for a given length in nm
        self.pts = lambda l: int(l * float(self.N) / self.system_length_nm)

        self.x_nm = np.linspace(-self.system_length_nm/2, \
            self.system_length_nm/2, self.N)
        
        meff = 0.067

        # bulk
        self.v_ev = self.pts(self.bulk_length_nm) * [self.w_h_ev]
        self.m_eff = self.pts(self.bulk_length_nm) * [meff]
        self.points_before = len(self.v_ev)
        # first barrier
        self.v_ev += self.pts(b_l) * [self.b_h_ev]
        self.m_eff += self.pts(b_l) * [meff]
        # well
        self.v_ev += self.pts(w_l) * [self.w_h_ev]
        self.m_eff += self.pts(w_l) * [meff]
        # second barrier
        self.v_ev += self.pts(b_l) * [self.b_h_ev]
        self.m_eff += self.pts(b_l) * [meff]
        self.points_after = len(self.v_ev)
        # span after second barrier
        self.v_ev += (self.N-len(self.v_ev)) * [self.w_h_ev]
        self.m_eff += (self.N-len(self.m_eff)) * [meff]

        # smooth the potential
        smooth_frac = int(float(self.N) / 500.0)
        self.v_ev = np.asarray([np.average(self.v_ev[max(0,i-smooth_frac):min(self.N-1,i+smooth_frac)]) for i in range(self.N)])

        # use numpy arrays
        self.m_eff = np.ones(self.N) * meff

        self.ajust_unities()

        # apply the bias
        if Fst != 0.0:
            # KV/cm
            self.Fst_v_cm = Fst * 1000.0
            self.Fst_v_m = 100.0 * self.Fst_v_cm
            self.Fst_j_m = self.Fst_v_m * self.q

            def Fst_shape(z):
                i = np.searchsorted(self.x_m, z)
                if i < self.points_before:
                    return 0.0
                elif self.points_before < i < self.points_after:
                    return (self.x_m[self.points_before] - z) * self.Fst_j_m
                else:
                    return -self.x_m[self.points_after] * self.Fst_j_m
            
            self.Fst_j = np.vectorize(Fst_shape)(self.x_m)
            
            self.Fst_ev = self.Fst_j / self.ev
            self.Fst_au = self.Fst_ev / self.au2ev
            self.v_ev += self.Fst_ev

            self.ajust_unities()

class QuantumWell(GenericPotential):
    def __init__(self, w_l, b_h, w_h, N=None, Fst=0.0, surround=1):
        super(QuantumWell,self).__init__()

        self.w_l_nm = w_l
        self.b_h_ev = b_h
        self.w_h_ev = w_h

        self.core_length_nm = w_l
        self.surround_times = surround # on each side

        self.system_length_nm = (2*self.surround_times + 1)*self.core_length_nm
        self.bulk_length_nm = (self.surround_times)*self.core_length_nm

        # this function return the number of points for a given length in nm
        self.pts = lambda l: int(l * float(self.N) / self.system_length_nm)

        self.x_nm = np.linspace(-self.system_length_nm/2, \
            self.system_length_nm/2, self.N)
        
        meff = 0.067

        # bulk
        self.v_ev = self.pts(self.bulk_length_nm) * [self.b_h_ev]
        self.m_eff = self.pts(self.bulk_length_nm) * [meff]
        self.points_before = len(self.v_ev)
        # well
        self.v_ev += self.pts(w_l) * [self.w_h_ev]
        self.m_eff += self.pts(w_l) * [meff]
        # span after second barrier
        self.v_ev += (self.N-len(self.v_ev)) * [self.b_h_ev]
        self.m_eff += (self.N-len(self.m_eff)) * [meff]

        # smooth the potential
        smooth_frac = int(float(self.N) / 500.0)
        self.v_ev = np.asarray([np.average(self.v_ev[max(0,i-smooth_frac):min(self.N-1,i+smooth_frac)]) for i in range(self.N)])

        # use numpy arrays
        self.m_eff = np.ones(self.N) * meff

        self.ajust_unities()

        # apply the bias
        if Fst != 0.0:
            # KV/cm
            self.Fst_v_cm = Fst * 1000.0
            self.Fst_v_m = 100.0 * self.Fst_v_cm
            self.Fst_j_m = self.Fst_v_m * self.q

            def Fst_shape(z):
                i = np.searchsorted(self.x_m, z)
                if i < self.points_before:
                    return 0.0
                elif self.points_before < i < self.points_after:
                    return (self.x_m[self.points_before] - z) * self.Fst_j_m
                else:
                    return -self.x_m[self.points_after] * self.Fst_j_m
            
            self.Fst_j = np.vectorize(Fst_shape)(self.x_m)
            
            self.Fst_ev = self.Fst_j / self.ev
            self.Fst_au = self.Fst_ev / self.au2ev
            self.v_ev += self.Fst_ev

            self.ajust_unities()

if __name__ == '__main__':
    #import numpy as np
    #import scipy.constants as cte
    #hbar=cte.value('Planck constant over 2 pi')
    #c=cte.value('speed of light in vacuum')
    #ev = cte.value('electron volt')
    #w=2*np.pi*c/(0.0000081)
    #f = np.vectorize(lambda n: hbar * w * (0.5+n) / ev)
    #print(f(range(3)))

    #system_properties = GenericPotential()
    #system_properties = MultiQuantumWell(w_n=2, total_length=150.0)
    #system_properties = FiniteQuantumWell(wh=25.0, wl=0.5)
    
    system_properties = BarriersWellSandwich(5.0, 4.0, 5.0, 0.4, 0.2, 0.0, Fst=0.0)
    #system_properties = DoubleBarrier(12., 10.0, 0.3, 0.0, Fst=0.0)
    #system_properties = QuantumWell(12.5, 1.6, 0.0, Fst=0.0, surround=2)
    #system_properties = BarriersWellSandwich(1.7, 0.0, 4.5, 1.0, 0.0, 0.0, Fst=0.0, surround=1)
    
    #################### EIGENSTATES ###########################################
    if True:
        result = system_properties.generate_eigenfunctions(1, steps=20000, verbose=True)
        np.savez('eigenfunctions/BarriersWellSandwichFstFdyn0', result['eigenstates'])
        print(result['eigenvalues'])
    else:
        files = np.load('eigenfunctions/BarriersWellSandwichFstFdyn0.npz')
        system_properties.states = files['arr_0']
        system_properties.values = np.zeros(system_properties.states.size, dtype=np.complex_)
        for i, state in enumerate(system_properties.states):
            system_properties.values[i] = system_properties.eigen_value(state)
    #################### EIGENSTATES ###########################################

    #################### PHOTOCURRENT ##########################################
    #pc = system_properties.photocurrent(energy=0.145, dt=5e-17)
    energies = np.linspace(0.1, 0.4, 100)
    photocurrent = []
    def get_pc(energy):
        pc = system_properties.photocurrent(energy=energy, dt=5e-17, Fdyn=5.0)
        #photocurrent.append(pc)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("[%s] > Energy: %.6f eV, PC: %.6e " % (now, energy, pc))
        return pc
    
    pool = Pool(processes=4)
    photocurrent = pool.map(get_pc, energies)
    plt.plot(energies, photocurrent)
    plt.show()
    np.savez('eigenfunctions/BarriersWellSandwichPhotoCurrent', photocurrent)
    #################### PHOTOCURRENT ##########################################

    #################### POTENTIAL SHAPE #######################################
    #potential_shape = system_properties.get_potential_shape()
    #plt.plot(potential_shape['x'], potential_shape['potential'])
    #plt.show()
    #################### POTENTIAL SHAPE #######################################
    
    

    #files = np.load('eigenfunctions/BarriersWellSandwich.npz')
    #system_properties.states = files['arr_0']
    #system_properties.values = np.zeros(system_properties.states.size, dtype=np.complex_)
    #for i, state in enumerate(system_properties.states):
    #    system_properties.values[i] = system_properties.eigen_value(state)
    
    #wave = 2*system_properties.states[0] + system_properties.states[1]
    #print(system_properties.wave_energy(wave))
    #print(4*system_properties.values[0] + system_properties.values[1])
    
    #result = system_properties.generate_eigenfunctions(10, steps=20000, verbose=True)
    #np.savez('eigenfunctions/BarriersWellSandwich', result['eigenstates'])
    #np.savez('eigenfunctions/BarriersWellSandwich', system_properties.states[0:3])
    
    ################ EVOLVE VISUAL #############################################
    #x = potential_shape['x']
    #x0 = x[0]+np.ptp(x)*0.3
    #delta_x = np.ptp(x) * 0.01
    #transmission = []
    #energies = np.linspace(0.0, 0.1, 20)
    #for energy in energies:
    #    pulse = system_properties.gaussian_pulse(delta_x=delta_x, x0=x0, E=energy)
    #    pulse, trans, bound = system_properties.evolve_pulse(display=True, steps=2000, dt=1e-17)
    ################ EVOLVE VISUAL #############################################

    ################ PULSE ANALYSIS ############################################
    #x = potential_shape['x']
    #x0 = x[0]+np.ptp(x)*0.3
    #delta_x = np.ptp(x) * 0.01
    #transmission = []
    #energies = np.linspace(0.0, 0.1, 20)
    #for energy in energies:
    #    pulse = system_properties.gaussian_pulse(delta_x=delta_x, x0=x0, E=energy)
    #    pk = fft(pulse)
    #    x_au = system_properties.x_au
    #    k_au = fftfreq(system_properties.N, d=(x_au[1]-x_au[0]))
    #    plt.plot(k_au, np.absolute(pk)**2)
    #    plt.show()
    ################ PULSE ANALYSIS ############################################

    #for energy in energies:
    #def evolve_all(energy):
    #    pulse = system_properties.gaussian_pulse(delta_x=delta_x, x0=x0, E=energy)
    #    pulse, trans, bound = system_properties.evolve_pulse(display=False, steps=20000)
    #    #transmission.append(trans)
    #    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #    print("[%s] > Energy: %.6f eV, trans: %.6f %%, bound: %.6f %%, refle: %.6f %%" % (now, energy, 100*trans, 100*bound, 100*(1.0-trans-bound)))
    #    return trans
    #
    #pool = Pool(processes=4)
    #transmission = pool.map(evolve_all, energies)
    #plt.plot(energies, transmission)
    #plt.show()
    #np.savez('eigenfunctions/BarriersWellSandwichTrans', transmission)

    #plt.plot(potential_shape['x'], potential_shape['potential'])
    #plt.plot(potential_shape['x'], potential_shape['m_eff'])
    #result = system_properties.generate_eigenfunctions(3, steps=20000)
    #for i, p in enumerate(result['eigenstates']):
    #    plt.plot(potential_shape['x'], (p*np.conjugate(p)).real+result['eigenvalues'][i])
    #print(result['eigenvalues'])
    #plt.show()