#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains classes for simulating generic potentials
"""

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from scipy.integrate import simps
import scipy.constants as cte
import scipy.special as sp
from scipy.sparse import diags
from scipy.linalg import inv
from scipy.signal import gaussian
from scipy.fftpack import fft, ifft, fftfreq
from types import LambdaType

import os, time, re

# very default values
DEFAULT_DT = 1e-19 # seconds
DEFAULT_N = 2048

class GenericPotential(object):
    """
    This class provides a basic framework for simulating an 
    heterostructure based on its building properties: layers 
    composition, materials used, sizes, room temperature

    It is possible to calculate:
    - eigenvalues and eigenfunctions
    - evolve waves in time under potential influence
    - calculate properties like photocurrent, transmission, reflextion

    The default potential is a Quantum Harmoic Oscillator for a wave 
    length of 8.1 µm

    The inputs must always be in:
    - Energy: `eV`
    - Length: `nm`
    - Time: `s`
    - Electric potential: `KV/cm`
    """

    def __init__(self, N=DEFAULT_N):
        """
        Parameters
        ----------
        N : int
            the number of grid points, it should be a power of 2

        Examples
        --------
        >>> from generic_potential import GenericPotential
        >>> device = GenericPotential(1024)
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

        # relations of interest
        self.au2ev = self.au_e / self.ev
        self.au2ang = self.au_l / 1e-10

        # specific for default quantum harmonic oscillator
        self.l = 0.0000081 # m
        self.f = self.c / self.l # Hz
        self.w = 2.0 * np.pi * self.f
        self.z_m = np.linspace(-5e-9,5e-9, self.N)
        self.v_j = 0.5 * self.me * self.z_m**2 * self.w**2
        self.z_nm = self.z_m / 1e-9
        self.v_ev = self.v_j / self.ev
        self.m_eff = np.ones(self.z_nm.size)

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
            a tuple consisting of a DataFrame with the eigenstate and 
            a float with the eigenvalue
        """
        return (self.device[['z_nm','state_{}'.format(n)]], \
            self.values[n])
    
    def get_system_states(self):
        """
        get a tuple with all the eigenstates and eigenvalues

        Returns
        -------
        info : tuple(DataFrame, array_like)
            a tuple consisting of a DataFrame with the eigenstates and 
            an array of float with the corresponding eigenstates
        """
        working = self._working_names()
        eigen = self._eigen_names()
        return (self.device[['z_nm']+eigen+working], self.values)

    def get_working(self, n):
        """
        return the nth wave function currently under analysis

        Parameters
        ----------
        n : integer
            the waves's index
        
        Returns
        -------
        wave : array_like
            a complex array with the systems wave **corresponding** to
            the system's length!!
        """
        return self.device['working_{}'.format(n)]

    def get_eigen_info(self, n):
        """
        return the nth (eigenfunction, eigenvalue) of the system

        Parameters
        ----------
        n : integer
            the eigenfunction's index
        
        Returns
        -------
        eigenfunction : tuple (DataFrame, float)
            a DataFrame with the systems eigenfunction 
            **corresponding** to the system's length!! and the 
            corresponding eigenvalue
        """
        return (self.device[['z_nm','state_{}'.format(n), 'm_eff']], \
            self.values[n])

    def get_potential_shape(self):
        """
        It return not only the potential, but also the spatial grid 
        and the effective masses along it

        Returns
        -------
        potential_info : dict
            the information about potential, spatial grid and effective 
            masses

        Examples
        --------
        >>> from specific_potentials import GenericPotential
        >>> generic = GenericPotential(1024)
        >>> generic.get_potential_shape()
        """
        return self.device[['z_nm', 'v_ev', 'm_eff']]

    # operations

    def calculate_eigenstates(self, n=1, precision=1e-4, dt=1e-19):
        """
        This function will generate the first `n` eigenvalues and
        eigenstates.

        Parameters
        ----------
        n : int
            the number of eigenstates/eigenvalues
        precision : float
            the minimum precision of the eigensvalues
        dt : float
            the time step in seconds
        
        Returns
        -------
        self : GenericPotential
            the current GenericPotential object for further use in 
            chain calls
        """
        # imaginary time
        self._set_dt(-1j * dt)

        # creates numpy arrays for hold the calculated values
        self.values = np.zeros(n, dtype=np.complex_)
        self.counters = np.zeros(n)

        # create kickstart states
        # they consist of legendre polinomials modulated by a 
        # gaussian
        short_grid = np.linspace(-1, 1, self.N)
        g = gaussian(self.N, std=int(self.N/100))
        states = np.array([g * sp.legendre(i)(short_grid) \
            for i in range(n)],dtype=np.complex_)

        for s in range(n):
            v_ant = 1.0
            while True:
                self.counters[s] += 1
                states[s] = self.evolution_operator(states[s])

                # Gram–Schmidt
                for m in range(s):
                    proj = simps(states[s] * \
                        states[m].conj(), self.z_au)
                    states[s] -= proj * states[m]

                # normalize
                states[s] /= np.sqrt(simps(np.abs(states[s])**2, \
                    self.z_au))

                if self.counters[s] % 1000 == 0:
                    self.values[s] = self._eigenvalue_tool(states[s])
                    print("%d: %.10e" % (s, self.values[s]))
                    if np.abs(1-self.values[s]/v_ant) < precision:
                        break
                    else:
                        v_ant = self.values[s]

        for i, state in enumerate(states):
            self.device['state_{}'.format(i)] = state

        return self

    def time_evolution(self, steps=2000, t0=0.0, \
        dt=None, imaginary=False, n=3, save=True, \
        load=True, verbose=False):
        """
        This function will evolve the `system_waves` in time. 
        It time is `imaginary`, then it will try to calculate the 
        `n` first eigenvalues and eigenstates of the system.

        Parameters
        ----------
        steps : integer
            the number of time steps to evolve the system
        t0 : float
            if the start time, useful when working with time dependent 
            potentials, it must be in seconds. **It will be used only 
            for real time evolution**
        dt : float
            the increment in time in **seconds** for each step, the 
            default is the system's default `dt` which is DEFAULT_DT.
        imaginary : boolean
            *False* stands for not imaginary time evolution, while 
            *True* stands for the opposite
        n : integer
            the number of eigenvalues and eigenstates to be calculated 
            in case of imaginary time evolution, the edfault is `3`
        save : boolean
            whether to save the results of an imaginary evolution, 
            which means save the eigenstates for further using. The 
            eigenstates will be saved in a file at the folder 
            `eigenfunction`
        load : boolean
            use stored eigenstates when available
        
        Returns
        -------
        self : GenericPotential
            the current GenericPotential object for further use in 
            chain calls
        """
        
        if imaginary:
            self._set_dt(dt * -1j)

            try:
                fst = self.bias_raw
            except:
                fst = 0.0

            try:
                fdyn = self.ep_dyn_raw
            except:
                fdyn = 0.0

            filename = "{cn}_{n}_{sts}_{bias:.2f}_{dyn:.2f}.csv"
            filename = filename.format(cn=self.__class__.__name__, \
                n=n, sts=steps, bias=fst, dyn=fdyn)
            directory = "devices"
            full_filename = os.path.join(directory, filename)

            if load:
                try:
                    device = pd.read_csv(full_filename)
                    complex_cols = [c for c in device.columns \
                        if re.match(r"^.*_\d+$", c)]
                    for c in complex_cols:
                        device[c] = device[c].str.replace('i','j'\
                            ).apply(lambda z: np.complex(z))
                    self.device = device
                    n = len(self._eigen_names())
                    self.values = [self._eigenvalue(i) for i in \
                        range(n)]
                    if verbose:
                        print('Using values from stored file')
                    return self
                except:
                    pass

            # creates numpy arrays for hold the calculated values
            self.values = np.zeros(n, dtype=np.complex_)

            # create kickstart states
            # they consist of legendre polinomials modulated by a 
            # gaussian
            short_grid = np.linspace(-1, 1, self.N)
            g = gaussian(self.N, std=int(self.N/100))
            states = np.array([g * sp.legendre(i)(short_grid) \
                for i in range(n)],dtype=np.complex_)
            for i, state in enumerate(states):
                self.device['state_{}'.format(i)] = state

            for s in range(n):
                sn = 'state_{}'.format(s)
                #for t in range(steps):
                for _ in range(steps):
                    self.device[sn] = \
                        self.evolution_operator(self.device[sn])

                    # gram-shimdt
                    for m in range(s):
                        sm = 'state_{}'.format(m)
                        proj = simps(self.device[sn] * \
                            np.conjugate(self.device[sm]), \
                            self.device.z_au)
                        self.device[sn] -= proj * self.device[sm]

                    # normalize
                    self.device[sn] /= np.sqrt(simps(self.device[sn] * \
                        np.conjugate(self.device[sn]), \
                            self.device.z_au))

                self.values[s] = self._eigenvalue(s)
                if verbose:
                    print('E_{0} = {1:.6f} eV'.format(s, \
                        self.values[s]))
            
            if save:
                if not os.path.exists(directory):
                    os.makedirs(directory)
                self.device.to_csv(full_filename)
        else:
            self._set_dt(dt)
            #t0_au = t0 / self.au_t
            # this might be parallel
            for w in self._working_names():
                #for t in range(steps):
                for _ in range(steps):
                    self.device[w] = \
                        self.evolution_operator(self.device[w])
                        #t0_au + t * self.dt_au)
        return self

    def normalize_device(self, method='pe', reset=False):
        """
        This function apply changes in the device structure or in 
        external conditions to the main `device` object

        Parameters
        ----------
        method : string
            the method for being used, which might be 
            Pseudo-Espectral ('pe'), Crank-Nicolson ('cn'), or 
            Runge-Kutta ('rk')
        reset : bool
            if True, the whole device is erased

        Returns
        -------
        self : GenericPotential
            the current GenericPotential object for further use in 
            chain calls
        """
        assert method in ['pe', 'cn', 'rk']

        try:
            if reset:
                raise Exception()
            device = self.device
        except:
            device = pd.DataFrame(dtype=np.complex_)
        
        # unique static inputs
        device['z_nm'] = np.copy(self.z_nm)
        device['v_ev'] = np.copy(self.v_ev)
        device['m_eff'] = np.copy(self.m_eff)

        # direct and reciprocal grids
        self.z_au = self.z_nm * 1e-9 / self.au_l # nm to au
        self.dz_au = self.z_au[1]-self.z_au[0]
        device['z_m'] = self.z_nm * 1e-9 # nm to m
        device['z_ang'] = self.z_nm * 10.0 # nm to ang
        device['z_au'] = np.copy(self.z_au)
        device['k_au'] = self.k_au = fftfreq(self.N, d=self.dz_au)

        # static potential (ti = time independent)
        self.v_au_ti = self.v_ev / self.au2ev
        device['v_j'] = self.v_ev * self.ev # ev to j
        device['v_au_ti'] = device['v_au'] = np.copy(self.v_au_ti)

        # check whether there is any bias to apply
        try:
            device['v_au_ti'] += self.bias_au
        except:
            pass

        # device is ready
        self.device = device

        # # check whether there is any dynamic field to apply
        # try:
        #     # v_au_td >> time dependent
        #     assert self.v_au_td and isinstance(self.v_au_td,LambdaType)
        #     self.v_au_full = lambda t: self.device.v_au_ti + \
        #         self.v_au_td(t)
        # except:
        #     self.v_au_full = lambda t: self.device.v_au_ti
        
        # runge-kutta 4th order
        if method == 'rk':
            def propagator(psi, t, tau):
                alpha = 1j / (2 * device['m_eff'] * self.dz_au ** 2)
                # beta = -1j * (self.v_au_full(t) + 1.0 / \
                #     (device['m_eff'] * self.dz_au ** 2))
                beta = -1j * (self.v_au_ti + 1.0 / \
                    (device['m_eff'] * self.dz_au ** 2))
                diagonal_1 = beta
                diagonal_2 = alpha[1:]
                diagonal_3 = alpha[:-1]
                diagonais = [diagonal_1, diagonal_2, diagonal_3]
                D = diags(diagonais, [0, 1, -1]).toarray()
                k1 = D.dot(psi)
                k2 = D.dot(psi + tau * self.dt_au * k1 / 2)
                k3 = D.dot(psi + tau * self.dt_au * k2 / 2)
                k4 = D.dot(psi + tau * self.dt_au * k3)
                return psi + tau * self.dt_au * \
                    (k1 + 2 * k2 + 2 * k3 + k4) / 6
            
            self.evolve_real = lambda psi, t: propagator(psi, t, 1.0)
            self.evolve_imag = lambda psi, t: propagator(psi, t, -1.0j)

        # crank-nicolson
        if method == 'cn':
            alpha_real = - self.dt_au * (1j / (2 * device['m_eff'] * self.dz_au ** 2))/2.0
            # beta_real = 1.0 - self.dt_au * (-1j * (self.v_au_full(0) + 1.0 / (device['m_eff'] * self.dz_au ** 2)))/2.0
            # gamma_real = 1.0 + self.dt_au * (-1j * (self.v_au_full(0) + 1.0 / (device['m_eff'] * self.dz_au ** 2)))/2.0
            beta_real = 1.0 - self.dt_au * (-1j * (self.v_au_ti + 1.0 / (device['m_eff'] * self.dz_au ** 2)))/2.0
            gamma_real = 1.0 + self.dt_au * (-1j * (self.v_au_ti + 1.0 / (device['m_eff'] * self.dz_au ** 2)))/2.0
            diagonal_1_real = beta_real
            diagonal_2_1_real = alpha_real[1:]
            diagonal_2_2_real = alpha_real[:-1]
            diagonais_real = [diagonal_1_real, diagonal_2_1_real, diagonal_2_2_real]
            invB_real = inv(diags(diagonais_real, [0, 1, -1]).toarray())
            diagonal_3_real = gamma_real
            diagonal_4_1_real = -alpha_real[1:]
            diagonal_4_2_real = -alpha_real[:-1]
            diagonais_2_real = [diagonal_3_real, diagonal_4_1_real, diagonal_4_2_real]
            C_real = diags(diagonais_2_real, [0, 1, -1]).toarray()
            D_real = invB_real.dot(C_real)

            tau = -1.0j
            alpha_imag = - tau * self.dt_au * (1j / (2 * device['m_eff'] * self.dz_au ** 2))/2.0
            # beta_imag = 1.0 - tau * self.dt_au * (-1j * (self.v_au_full(0) + 1.0 / (device['m_eff'] * self.dz_au ** 2)))/2.0
            # gamma_imag = 1.0 + tau * self.dt_au * (-1j * (self.v_au_full(0) + 1.0 / (device['m_eff'] * self.dz_au ** 2)))/2.0
            beta_imag = 1.0 - tau * self.dt_au * (-1j * (self.v_au_ti + 1.0 / (device['m_eff'] * self.dz_au ** 2)))/2.0
            gamma_imag = 1.0 + tau * self.dt_au * (-1j * (self.v_au_ti + 1.0 / (device['m_eff'] * self.dz_au ** 2)))/2.0
            diagonal_1_imag = beta_imag
            diagonal_2_1_imag = alpha_imag[1:]
            diagonal_2_2_imag = alpha_imag[:-1]
            diagonais_imag = [diagonal_1_imag, diagonal_2_1_imag, diagonal_2_2_imag]
            invB_imag = inv(diags(diagonais_imag, [0, 1, -1]).toarray())
            diagonal_3_imag = gamma_imag
            diagonal_4_1_imag = -alpha_imag[1:]
            diagonal_4_2_imag = -alpha_imag[:-1]
            diagonais_2_imag = [diagonal_3_imag, diagonal_4_1_imag, diagonal_4_2_imag]
            C_imag = diags(diagonais_2_imag, [0, 1, -1]).toarray()
            D_imag = invB_imag.dot(C_imag)
            
            self.evolve_real = lambda psi, t: D_real.dot(psi)
            self.evolve_imag = lambda psi, t: D_imag.dot(psi)

        # pseudo-espectral
        if method == 'pe':
            exp_v2 = np.exp(- 0.5j * self.v_au_ti * self.dt_au)
            exp_t = np.exp(- 0.5j * (2 * np.pi * self.k_au) ** 2 \
                * self.dt_au / self.m_eff)
            self.evolution_operator = lambda psi: exp_v2 * ifft(exp_t \
                * fft(exp_v2 * psi))

        return self

    def turn_bias_on(self, bias, core_only=False):
        """
        this function applies a static bias accross the system, the 
        `bias` must be given in KV/cm, God knows why...

        if the `core_only` is true, the bias is not applied to the span 
        that surrounds the system under study

        Parameters
        ----------
        bias : float
            the bias in KV/cm
        core_only : boolean
            whether to apply the bias in the whole system or only in 
            the core under study and not in the span/bulk area

        Returns
        -------
        self : GenericPotential
            the current GenericPotential object for further use in 
            chain calls
        """
        self.bias_raw = bias
        self.bias_v_cm = bias * 1000.0
        self.bias_v_m = 100.0 * self.bias_v_cm
        self.bias_j_m = self.bias_v_m * self.q

        if core_only:
            def bias_shape(z):
                i = np.searchsorted(self.device.z_m, z)
                if i < self.points_before:
                    return 0.0
                elif self.points_before < i < self.points_after:
                    return (self.device.z_m[self.points_before] - z) * \
                        self.bias_j_m
                else:
                    return -self.device.z_m[self.points_after] * \
                        self.bias_j_m
            self.bias_j = np.vectorize(bias_shape)(self.device.z_m)
        else:
            self.bias_j = np.vectorize(lambda z: \
                (self.device.z_m[0]-z)*self.bias_j_m)(self.device.z_m)
        
        self.bias_ev = self.bias_j / self.ev
        self.bias_au = self.bias_ev / self.au2ev

        return self.normalize_device()

    def turn_bias_off(self):
        """
        this function removes the bias previously applied if any...

        Returns
        -------
        self : GenericPotential
            the current GenericPotential object for further use in 
            chain calls
        """
        self.bias_au = None
        return self.normalize_device()

    def turn_dyn_on(self, ep_dyn, w_len=8.1e-6, f=None, \
        energy=None, core_only=False):
        """
        this function applies a sine wave like an electric field to 
        the system

        if the `core_only` is true, the bias is not applied to the 
        span that surrounds the system under study

        Parameters
        ----------
        ep_dyn : float
            the electric potential in KV/cm
        w_len : float
            the electric field wave length in meters
        f : float
            the electric field frequency in Hz
        energy : float
            the wave's energy in eV where it is going to be used 
            `E = hbar * w`
        core_only : boolean
            whether to apply the bias in the whole system or only in 
            the core under study and not in the span/bulk area

        Returns
        -------
        self : GenericPotential
            the current GenericPotential object for further use in 
            chain calls
        """
        self.ep_dyn_raw = ep_dyn

        # KV/cm
        self.ep_dyn_v_cm = ep_dyn * 1000.0
        self.ep_dyn_v_m = 100.0 * self.ep_dyn_v_cm
        self.ep_dyn_j_m = self.ep_dyn_v_m * self.q
        self.ep_dyn_j = np.vectorize(lambda z: \
            (self.device.z_m[0]-z) * self.ep_dyn_j_m)(self.device.z_m)
        self.ep_dyn_ev = self.ep_dyn_j / self.ev
        self.ep_dyn_au = self.ep_dyn_ev / self.au2ev

        if energy:
            self.omega_au = (energy / self.au2ev) / self.hbar_au
        elif f:
            self.omega_au = 2.0 * np.pi * (f * self.au_t)
        elif w_len:
            f = self.c / w_len # Hz
            self.omega_au = 2.0 * np.pi * (f * self.au_t)
        else:
            raise Exception("""It must be informed one of the following: 
                wave energy, wave frequency, or wave length""")
        
        self.v_au_td = lambda t: self.ep_dyn_au*np.sin(self.omega_au*t)
        return self.normalize_device()

    def turn_dyn_off(self):
        """
        this function removes the radiation previously applied if any...

        Returns
        -------
        self : GenericPotential
            the current GenericPotential object for further use in 
            chain calls
        """
        self.v_au_td = None
        return self.normalize_device()

    def work_on(self, n=0, indexes=None):
        """
        set some eigenfunction or some of them to the working waves

        Parameters
        ----------
        n : integer
            the index of some eigenfunction to be used as system wave
        indexes : array_like
            the indexes of some eigenfunctions to be used as system wave

        Returns
        -------
        self : GenericPotential
            the current GenericPotential object for further use in 
            chain calls
        """

        # erase working waves
        self.device.drop(self._working_names(), axis=1, \
            inplace=True, errors='ignore')

        if indexes:
            for i, idx in enumerate(indexes):
                self.device['working_{}'.format(i)] = \
                    self.device['state_{}'.format(idx)] 
            #self.working_waves = self.states.take(indexes)
        else:
            #self.working_waves = np.array([np.copy(self.states[n])])
            self.device['working_{}'.format(n)] = \
                self.device['state_{}'.format(n)] 
        return self

    # internals

    def _set_dt(self, dt=None):
        """
        this function might be used for setting the default time 
        increment if *None* is given, then it will assume the value 
        of `DEFAULT_DT`

        Parameters
        ----------
        dt : float
            the default time increment, in **seconds**

        Returns
        -------
        self : GenericPotential
            the current GenericPotential object for further use in 
            chain calls
        """
        self.dt = dt or DEFAULT_DT
        self.dt_au = self.dt / self.au_t

        return self
    
    def _eigenvalue_tool(self, eigenstate, A2=None):
        """ 
        calculate the eigenvalue of a given state under the device
        conditions, **caution** it only makes sense for a eigenvalue
        of the system's hamiltonian

        Patameters
        ----------
        eigenstate : array_like
            an eigenstate for calculating the eigenvalue
        A2 : float
            if by any reason it is available, it would help a lot
                A2 = <eigenstate|eigenstate>

        Returns
        -------
        eigenvalue : float
            the eigenvalue corresponding to the indicated eigenstate
        """
        # get only necessary columns
        if not A2:
            A2 = simps(np.abs(eigenstate)**2, self.z_au)
        sec_derivative = (eigenstate[:-2] - 2 * eigenstate[1:-1] + \
            eigenstate[2:]) / self.dz_au**2
        psi = eigenstate[1:-1]
        # <Psi|H|Psi>
        p_h_p = simps(psi.conj() * (-0.5 * sec_derivative \
            / self.m_eff[1:-1] + self.v_au_ti[1:-1] * psi), \
            self.z_au[1:-1])
        # divide by <Psi|Psi> 
        p_h_p /= A2
        return p_h_p.real * self.au2ev # eV

    def _eigenvalue(self, n, t=0.0):
        """ 
        **Only** for eigenfunctions of the current device

        Patameters
        ----------
        n : integer
            the eigenstate index
        t : float
            the time in atomic units, for use in time dependent 
            potentials

        Returns
        -------
        eigenvalue : float
            the eigenvalue corresponding to the indicated eigenstate
        """
        sn = "state_{}".format(n)
        sn_st = "state_{}_conjugate".format(n)
        sn_d2 = "state_{}_2nd_derivative".format(n)
        
        # get only necessary columns
        device = self.device[['z_au', 'm_eff', sn]]

        # second derivative of psi_n
        device[sn_d2] = (device[sn].shift(1) - 2 * device[sn] + \
            device[sn].shift(-1)) / device['z_au'].diff()**2
        
        # remove those NA in the edges
        device = device.iloc[1:-1]

        # complex conjugate
        device[sn_st] = np.conjugate(device[sn])

        # <Psi|H|Psi>
        p_h_p = simps(device[sn_st] * (-0.5 * device[sn_d2] / \
            #device['m_eff'] + self.v_au_full(t)[1:-1] * device[sn]), \
            device['m_eff'] + self.v_au_ti[1:-1] * device[sn]), \
            device['z_au'])
        # / <Psi|Psi> because I trust no one
        p_h_p /= simps(device[sn_st] * device[sn], device['z_au'])

        return p_h_p.real * self.au2ev # return value in eV

    def _eigen_names(self):
        """
        it returns the name of the columns where the eigenstates are 
        stored in the main device

        Returns
        -------
        names : array_like
            the names of the columns in the device where eigenstates 
            are stored
        """
        cols = self.device.columns
        return sorted([c for c in cols if re.match(r"^state_\d+$", c)])

    def _working_names(self):
        """
        it returns the name of the columns where the working waves are 
        stored in  the main device

        Returns
        -------
        names : array_like
            the names of the columns in the device where working waves 
            are stored
        """
        cols = self.device.columns
        return sorted([c for c in cols if \
            re.match(r"^working_\d+$", c)])

    # miscellaneous and legacy

    def photocurrent(self, energy, T=1.0e-12, ep_dyn=5.0, dt=None):
        """
        this function calculates the photocurrent *************

        Parameters
        ----------
        energy : float
            the energy of incident photons in eV
        T : float
            the total time for measuring the electric current 
            in seconds
        ep_dyn : float
            the intensity of the 

        Returns
        -------
        j : float
            the photocurrent in Ampere (not sure hehe)
        """
        self._set_dt(dt).turn_dyn_on(ep_dyn=ep_dyn, energy=energy)
        
        T_au = T / self.au_t
        t_grid_au = np.linspace(0.0, T_au, int(T_au / self.dt_au))

        pb = self.points_before - 100
        pa = self.points_after + 100

        psi = self.device.state_0
        j_t = []
        
        psi = self.work_on(0).get_working(0)

        for t_au in t_grid_au:
            psi = self.evolve_real(psi, t=t_au)
            j_l = ((-0.5j/(self.device.m_eff[pb])) * \
                (psi[pb].conjugate() * \
                (psi[pb+1]-psi[pb-1])-psi[pb]*(psi[pb+1].conjugate()-\
                psi[pb-1].conjugate())) / (2*self.dz_au)).real
            j_r = ((-0.5j/(self.device.m_eff[pa])) * \
                (psi[pa].conjugate() * \
                (psi[pa+1]-psi[pa-1])-psi[pa]*(psi[pa+1].conjugate()-\
                psi[pa-1].conjugate())) / (2*self.dz_au)).real
            j_t.append(j_r-j_l)
            
        return self.q * (simps(j_t, t_grid_au) / T_au) / T

    def wave_energy(self, psi):
        """
        Calculates the energy of an arbitrary wave in the system
        it depends on how many eigenvalues/eigenfunctions are already 
        calculated since it is going to be a superposition

        Parameters
        ----------
        psi : array_like
            an arbitrary wave, fitting the system's size (number of 
            points) and corresponding to its spatial grid

        Returns
        -------
        energ : float
            the energy of the given wave in the current system
        """
        energy = 0.0
        for i, value in enumerate(self.values):
            state = self.device['state_{}'.format(i)]
            state_st = state.conjugate()
            an = simps(state_st * psi, self.device.z_au) / \
                simps(state_st * state, self.device.z_au)
            energy += (an.conjugate()*an)*value
        return energy
