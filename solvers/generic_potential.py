#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains classes for simulating generic potentials
"""

import os
import time
import re
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

from scipy                  import constants as cte
from scipy.integrate        import simps
from scipy.sparse           import diags
from scipy.linalg           import inv
from scipy.signal           import gaussian
from scipy.special          import legendre, expit
from scipy.fftpack          import fft, ifft, fftfreq
from scipy.spatial.distance import cdist

# very default values
DEFAULT_DT = 1e-19 # seconds
DEFAULT_N  = 2048

class GenericPotential(object):
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
        self.au_l    = cte.value('atomic unit of length')
        self.au_t    = cte.value('atomic unit of time')
        self.au_e    = cte.value('atomic unit of energy')
        self.au_v    = cte.value('atomic unit of electric potential')
        self.hbar_au = 1.0
        self.me_au   = 1.0

        # other useful constants
        self.ev = cte.value('electron volt')
        self.c  = cte.value('speed of light in vacuum') # m/s
        self.me = cte.value('electron mass')
        self.q  = cte.value('elementary charge')

        # relations of interest
        self.au2ev  = self.au_e / self.ev
        self.au2ang = self.au_l / 1e-10

        # specific for default quantum harmonic oscillator
        self.l     = 0.0000081 # m
        self.f     = self.c / self.l # Hz
        self.w     = 2.0 * np.pi * self.f
        self.z_m   = np.linspace(-5e-9,5e-9, self.N)
        self.v_j   = 0.5 * self.me * self.z_m**2 * self.w**2
        self.z_nm  = self.z_m / 1e-9
        self.v_ev  = self.v_j / self.ev
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
        return (self.device[['z_nm','state_{}'.format(n)]], self.values[n])
    
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
        eigen   = self._eigen_names()
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
        It return not only the potential, but also the spatial grid and the 
        effective masses along it

        Returns
        -------
        potential_info : dict
            the information about potential, spatial grid and effective masses

        Examples
        --------
        >>> from specific_potentials import GenericPotential
        >>> generic = GenericPotential(1024)
        >>> generic.get_potential_shape()
        """
        return self.device[['z_nm', 'v_ev', 'm_eff']]

    # almost independent

    def orthonormal(self, n=2, size=2048):
        """Return the first `n` orthonormal legendre polynoms weighted by a 
        gaussian. They are useful as kickstart arrays for a imaginary time 
        evolution
        
        Params
        ------
        n : int
            the number of vectors (default: {2})
        size : int
            the size of each vector (default: {2048})
        """
        sg  = np.linspace(-1, 1, size) # short grid
        g   = gaussian(size, std=int(size/100)) # gaussian
        vls = [g*legendre(i)(sg) for i in range(n)]
        return np.array(vls, dtype=np.complex_)

    def fi(self, v, i):
        """ short for flex_index, find the index `i` in array `v`
        `i` might be float, so that for `i = 1.5`, the return would be
        `(v[2]+v[1])/2`
        
        Params
        ------
        v : array_like
            a list of real or complex numbers
        i : int
            a semi integer number between 0 and the vector's `v` length
        """
        if i < 0.0:
            return v[0]
        elif i >= len(v) - 1:
            return v[-1]
        
        try:
            i_up   = int(np.ceil(i))
            i_down = int(np.floor(i))
            return (v[i_up]+v[i_down])/2.0
        except:
            return 0.0

    def get_break(self, x, y, n=10):
        """finds where the deriative of y(x) changes more than 5% 
        since we deal with many wells and barries, it helps to find
        where they begin
        
        Params
        ------
        x : array_like
            an array with the x values
        y : array_like
            an array with the y values
        n : int
            the number of points to ignore in the borders
        """
        der_y = np.array(y[2:]-y[:-2])/np.array(x[2:]-x[:-2])
        for i in range(n, len(der_y)):
            last_n  = np.average(der_y[i-n:i-1])
            if last_n == 0 and der_y[i] != 0 \
                or last_n != 0 and np.abs(der_y[i]/last_n-1) > 0.05:
                return i
        return int(len(y)/3)

    def eigenvalue(self, z, V, psi, m):
        """calculate eigenvalue like E=<Psi|H|Psi>/<Psi|Psi>
        where H = T + `V`, T is the only kinect operator in one dimension

        Params
        ------
        z : array_like
            the coordinates grid
        V : array_like
            the potential
        psi : array_like
            the wave function
        m : array_like
            the (position dependent) effective mass
        """
        N     = self.N
        fi    = self.fi
        dz    = np.append(z[1:]-z[:-1], z[1]-z[0])
        dz2   = dz**2
        h_psi = np.zeros(N, dtype=np.complex_)
        
        for i in range(N):
            h_psi[i] = ((0.5/dz2[i])*(1.0/fi(m,i+0.5)+\
                        1.0/fi(m,i-0.5))+V[i])*psi[i]
            if i > 0:
                h_psi[i] += -(0.5/dz2[i])*(psi[i-1]/fi(m,i-0.5))
            if i < N-1:
                h_psi[i] += -(0.5/dz2[i])*(psi[i+1]/fi(m,i+0.5))
                
        psi_h_psi = simps(psi.conj()*h_psi, z)
        return (psi_h_psi / simps(psi.conj()*psi, z)).real

    def bound_states(self, z, V, m, nmax=20, precision=1e-9, verbose=False):
        """ find the bound eigenstates for a given potential `V` under the
        effective mass approximation (`m`). It uses the inverse interaction
        method. It is possible to use a z grid with varying step size,
        but the increase in time is huge

        Params
        ------
        z : array_like
            the coordinates grid
        V : array_like
            the potential
        m : array_like
            the (position dependent) effective mass
        nmax : int
            since it uses the inverse interaction, it is the max number of
            kick start eigenvalues in a grid between `min(V)` and `max(V)`
        precision : float
            it is the max error allowed for each eigenvalue, the error is just
            (new_eigenalue_n - old_eigenalue_n)/old_eigenalue_n
        verbose : boolean
            in case of `True`, logs and messages about the process are going
            to be printed on the screen

        """
        N        = self.N
        fi       = self.fi
        forecast = np.linspace(np.min(V), np.max(V), nmax)
        dz       = np.append(z[1:]-z[:-1], z[1]-z[0])
        dz2      = dz**2
        
        # kick start eigenstates
        eigenstates = self.orthonormal(nmax, size=N)
        eigenvalues = np.zeros(nmax)
        
        # matrix diagonals
        sub_diag  = np.zeros(N-1, dtype=np.complex_)
        main_diag = np.zeros(N  , dtype=np.complex_)

        def get_invA(v_shift=0.0):
            """Applies a shift in the potential, same as H'=H-beta """
            for i in range(N):
                try:
                    main_diag[i] = (0.5/dz2[i])*(1.0/fi(m,i+0.5)+\
                                    1.0/fi(m,i-0.5))+(V[i]-v_shift)
                except:
                    main_diag[i] = 0.0

                if i < N-1:
                    sub_diag[i] = -(0.5/dz2[i])*(1.0/fi(m,i+0.5))

            diagonals = [main_diag, sub_diag, sub_diag]
            A         = diags(diagonals, [0, -1, 1]).toarray()
            return inv(A)
                
        counters            = np.zeros(nmax)
        timers              = np.zeros(nmax)
        precisions          = np.zeros(nmax)
        vectors_sqeuclidean = np.zeros(nmax)
        
        for s in range(nmax):
            last_ev = 1.0
            last_es = np.zeros(N, dtype=np.complex_)
            
            shift     = forecast[s]
            invA      = get_invA(shift)
            V_shifted = V-shift
            
            while True:
                start_time = time.time()
                eigenstates[s] = invA.dot(eigenstates[s])
                counters[s] += 1

                # normalize
                A = np.sqrt(simps(eigenstates[s]*eigenstates[s].conj(), z))
                eigenstates[s] /= A
                timers[s] += time.time() - start_time

                eigenvalues[s] = self.eigenvalue(z,V_shifted,eigenstates[s],m) \
                                    + shift

                # check precision
                precisions[s] = np.abs(1-eigenvalues[s]/last_ev)
                last_ev = eigenvalues[s]

                if precisions[s] < precision:
                    XA = [np.abs(eigenstates[s])**2]
                    XB = [np.abs(last_es)**2]
                    vectors_sqeuclidean[s] = cdist(XA, XB, 'sqeuclidean')[0][0]
                    break

                last_es = np.copy(eigenstates[s])

            if verbose:
                print("[{0}/{1}] ready! E = {2:.6f} eV".format(s+1, nmax, \
                    eigenvalues[s]*self.au2ev))
        
        sort_index  = eigenvalues.argsort()
        eigenvalues = eigenvalues[sort_index]
        eigenstates = eigenstates[sort_index]
        
        iz_left     = self.get_break(z, V)
        iz_right    = len(V)-self.get_break(z, V[::-1])
        golden_ones = [0]
        
        for i in range(eigenvalues.size):
            # drop repeated and unbounded states
            if i == 0 or np.abs(eigenvalues[i]/eigenvalues[i-1]-1) < 0.1 \
                or eigenvalues[i] > np.max(V):
                continue
            
            # drop not confined states
            state     = eigenstates[i].copy()
            state_l   = state[:iz_left]
            state_m   = state[iz_left:iz_right]
            state_r   = state[iz_right:]
            int_left  = simps(state_l*state_l.conj(), z[:iz_left]).real
            int_mid   = simps(state_m*state_m.conj(), z[iz_left:iz_right]).real
            int_right = simps(state_r*state_r.conj(), z[iz_right:]).real
            
            if int_left+int_right > int_mid:
                continue
            
            golden_ones.append(i)
            
        return {
            'eigenvalues': eigenvalues[golden_ones],
            'eigenstates': eigenstates[golden_ones],
            'counters': counters[golden_ones],
            'timers': timers[golden_ones],
            'precisions': precisions[golden_ones],
            'squared_euclidean_dist': vectors_sqeuclidean[golden_ones]
        }

    def time_evolution_operator(self, psi, t, dt):
        """this function make a transtion in time for psi, from t to t+dt
        the Hamiltonian is that of T+V(t)

        Params
        ------
        psi : array_like
            the state (in atomic units) to evolve in time
        t : float
            the initial time in atomic units
        dt : float
            the time increment in atomic units
        """

        # evolution operator
        exp_v2 = np.exp(- 0.5j * (self.v_au_ti+self.v_au_td(t)) * dt)
        exp_t  = np.exp(- 0.5j * (2 * np.pi * self.k_au) ** 2 * dt / self.m_eff)
        return exp_v2 * ifft(exp_t * fft(exp_v2 * psi))


    def solve_eigen_problem(self, save=True, load=True, verbose=False, nmax=10):
        """This function will calculate eigenvectors and eigenvalues,
        properly setting the results in the device's dataframe        

        Parameters
        ----------
        save : boolean
            whether to save the results, which means save the eigenstates for 
            further using. The eigenstates will be saved in a file at the folder 
            `devices`
        load : boolean
            use stored eigenstates when available
        nmax : integer
            it is the maximum of eigenvalues/eigenvectors to calculate, default
            is `10`, but it will be less then ten in most cases
        verbose : boolean
            in case of `True`, logs and messages about the process are going
            to be printed on the screen

        Returns
        -------
        self : GenericPotential
            the current GenericPotential object for further use in chain calls
        """
        
        try:
            fst = self.bias_raw
        except:
            fst = 0.0

        try:
            fdyn = self.ep_dyn_raw
        except:
            fdyn = 0.0

        filename      = "{cn}_{n}_{bias:.2f}_{dyn:.2f}.csv"
        filename      = filename.format(cn=self.__class__.__name__, 
                        n=nmax, bias=fst, dyn=fdyn)
        directory     = "devices"
        full_filename = os.path.join(directory, filename)

        if load:
            try:
                device       = pd.read_csv(full_filename)
                complex_cols = [c for c in device.columns \
                    if re.match(r"^.*_\d+$", c)]

                for c in complex_cols:
                    device[c] = device[c].str.replace('i','j'\
                        ).apply(lambda z: np.complex(z))

                self.device = device
                n           = len(self._eigen_names())
                self.values = [self._eigenvalue(i) for i in range(n)]
                if verbose:
                    print('Using values from stored file')
                return self
            except:
                pass

        
        # calculated bound states
        bound = self.bound_states(self.z_au, self.device['v_au_ti'].values, 
            self.device['m_eff'].values, nmax=max(20, nmax), verbose=verbose)

        self.values = bound['eigenvalues'] * self.au2ev
        for i, state in enumerate(bound['eigenstates']):
            # normalize to nm
            nc = np.sqrt(simps(state*state.conj(), self.z_nm))
            self.device['state_{}'.format(i)] = state / nc
        
        if save:
            if not os.path.exists(directory):
                os.makedirs(directory)
            self.device.to_csv(full_filename)
    
        return self

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
            the current GenericPotential object for further use in chain calls
        """
        # imaginary time
        self._set_dt(-1j * dt)
        self.normalize_device()

        # creates numpy arrays for hold the calculated values
        self.values   = np.zeros(n, dtype=np.complex_)
        self.counters = np.zeros(n)

        # create kickstart states
        # they consist of legendre polinomials modulated by a 
        # gaussian
        short_grid = np.linspace(-1, 1, self.N)
        g          = gaussian(self.N, std=int(self.N/100))
        states     = np.array([g * legendre(i)(short_grid) \
            for i in range(n)],dtype=np.complex_)

        for s in range(n):
            v_ant = 1.0
            while True:
                self.counters[s] += 1
                states[s] = self.evolution_operator(states[s])

                # Gram–Schmidt
                for m in range(s):
                    proj       = simps(states[s] * states[m].conj(), self.z_au)
                    states[s] -= proj * states[m]

                # normalize
                states[s] /= np.sqrt(simps(np.abs(states[s])**2, self.z_au))

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

    def time_evolution(self, steps=2000, t0=0.0, dt=None, imaginary=False, \
        n=3, save=True, load=True, verbose=False):
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
            self._set_dt((dt or DEFAULT_DT) * -1j)

            try:
                fst = self.bias_raw
            except:
                fst = 0.0

            try:
                fdyn = self.ep_dyn_raw
            except:
                fdyn = 0.0

            filename      = "{cn}_{n}_{sts}_{bias:.2f}_{dyn:.2f}.csv"
            filename      = filename.format(cn=self.__class__.__name__, 
                            n=n, sts=steps, bias=fst, dyn=fdyn)
            directory     = "devices"
            full_filename = os.path.join(directory, filename)

            if load:
                try:
                    device       = pd.read_csv(full_filename)
                    complex_cols = [c for c in device.columns \
                        if re.match(r"^.*_\d+$", c)]

                    for c in complex_cols:
                        device[c] = device[c].str.replace('i','j'\
                            ).apply(lambda z: np.complex(z))

                    self.device = device
                    n           = len(self._eigen_names())
                    self.values = [self._eigenvalue(i) for i in range(n)]
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
            g          = gaussian(self.N, std=int(self.N/100))
            states     = np.array([g * legendre(i)(short_grid) \
                            for i in range(n)],dtype=np.complex_)
            
            for i, state in enumerate(states):
                self.device['state_{}'.format(i)] = state

            for s in range(n):
                sn = 'state_{}'.format(s)
                #for t in range(steps):
                for _ in range(steps):
                    self.device[sn] = self.evolution_operator(self.device[sn])

                    # gram-shimdt
                    for m in range(s):
                        sm   = 'state_{}'.format(m)
                        proj = simps(self.device[sn] * \
                            np.conjugate(self.device[sm]), self.device.z_au)
                        
                        self.device[sn] -= proj * self.device[sm]

                    # normalize
                    self.device[sn] /= np.sqrt(simps(self.device[sn] * \
                        np.conjugate(self.device[sn]), self.device.z_au))

                self.values[s] = self._eigenvalue(s)
                if verbose:
                    print('E_{0} = {1:.6f} eV'.format(s, self.values[s]))
            
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
                    self.device[w] = self.evolution_operator(self.device[w])
        return self

    def normalize_device(self, reset=False):
        """
        This function apply changes in the device structure or in 
        external conditions to the main `device` object

        Parameters
        ----------
        reset : bool
            if True, the whole device is erased

        Returns
        -------
        self : GenericPotential
            the current GenericPotential object for further use in 
            chain calls
        """
        
        try:
            if reset:
                raise Exception()
            device = self.device
        except:
            device = pd.DataFrame(dtype=np.complex_)
        
        # unique static inputs
        device['z_nm']  = np.copy(self.z_nm)
        device['v_ev']  = np.copy(self.v_ev)
        device['m_eff'] = np.copy(self.m_eff)

        # direct and reciprocal grids
        self.z_au       = self.z_nm * 1e-9 / self.au_l # nm to au
        self.z_ang      = self.z_nm * 10.0 # nm to ang
        self.dz_au      = self.z_au[1]-self.z_au[0]
        device['z_m']   = self.z_nm * 1e-9 # nm to m
        device['z_ang'] = self.z_nm * 10.0 # nm to ang
        device['z_au']  = np.copy(self.z_au)
        device['k_au']  = self.k_au = fftfreq(self.N, d=self.dz_au)

        # static potential (ti = time independent)
        self.v_au_ti      = self.v_ev / self.au2ev
        device['v_j']     = self.v_ev * self.ev # ev to j
        device['v_au_ti'] = device['v_au'] = np.copy(self.v_au_ti)

        # check whether there is any bias to apply
        try:
            device['v_au_ti'] += self.bias_au
        except:
            pass

        # device is ready
        self.device = device

        # evolution operator
        exp_v2 = np.exp(- 0.5j * self.v_au_ti * self.dt_au)
        exp_t  = np.exp(- 0.5j * (2 * np.pi * self.k_au) ** 2 \
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
        N              = self.N
        self.bias_raw  = bias
        self.bias_v_cm = bias * 1e3
        self.bias_v_m  = 1e2 * self.bias_v_cm
        self.bias_j_m  = self.bias_v_m * self.q

        self.points_before = 0
        self.points_after  = self.N-1
        
        if core_only:
            self.points_before = self.get_break(self.z_nm, self.v_ev)
            self.points_after  = N-1-self.get_break(self.z_nm, self.v_ev[::-1])

        device_border_left_ang  = self.z_ang[self.points_before]
        device_border_right_ang = self.z_ang[self.points_after]
        Vst_j                   = lambda z: -(z*1e-10)*(self.bias_j_m)
        V_left                  = Vst_j(device_border_left_ang)
        V_right                 = Vst_j(device_border_right_ang)

        def find_bias(z):
            if z <= device_border_left_ang:
                return V_left
            elif z >= device_border_right_ang:
                return V_right
            return Vst_j(z)

        self.bias_j  = np.vectorize(find_bias)(self.z_ang)
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
        self.ep_dyn_v_m  = 100.0 * self.ep_dyn_v_cm
        self.ep_dyn_j_m  = self.ep_dyn_v_m * self.q
        self.ep_dyn_j    = np.vectorize(lambda z: \
            (self.device.z_m[0]-z) * self.ep_dyn_j_m)(self.device.z_m)
        self.ep_dyn_ev   = self.ep_dyn_j / self.ev
        self.ep_dyn_au   = self.ep_dyn_ev / self.au2ev

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
        self.dt    = dt or DEFAULT_DT
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
        psi            = eigenstate[1:-1]
        # <Psi|H|Psi>
        p_h_p          = simps(psi.conj() * (-0.5 * sec_derivative \
                        / self.m_eff[1:-1] + self.v_au_ti[1:-1] * psi), \
                        self.z_au[1:-1])
        # divide by <Psi|Psi> 
        p_h_p         /= A2
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
        sn    = "state_{}".format(n)
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

    def photocurrent(self, energy, T=1e-12, ep_dyn=5.0, dt=None, verbose=False):
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
        
        T_au      = T / self.au_t
        t_grid_au = np.linspace(0.0, T_au, int(T_au / self.dt_au))
        pb        = self.points_before - 100
        pa        = self.points_after + 100
        psi       = np.array(self.device.state_0, dtype=np.complex_)
        psi      /= np.sqrt(simps(psi*psi.conj(), self.z_au))
        meff      = self.m_eff
        z_au      = self.z_au
        i         = 0
        j_t       = []
        
        L = np.ptp(self.z_ang)
        N = self.N
        z = np.linspace(-L/2,L/2,N)
        killer = np.array([min(l,r) for l,r in zip(expit((400-z)/10), expit((z+400)/10))], dtype=np.complex_)
        
        for t_au in t_grid_au:
            i += 1
            if verbose and i % 1000 == 0:
                print("[{0:.4e}/{1:.4e}] seg".format(t_au * self.au_t, T))

            psi = self.time_evolution_operator(psi, t_au, self.dt_au)*killer
            

            j_l = ((-0.5j/(meff[pb])) * (psi[pb].conj() * (psi[pb+1]-psi[pb-1])-psi[pb]*(psi[pb+1].conj()-psi[pb-1].conj())) / (z_au[pb+1]-z_au[pb-1])).real
            j_r = ((-0.5j/(meff[pa])) * (psi[pa].conj() * (psi[pa+1]-psi[pa-1])-psi[pa]*(psi[pa+1].conj()-psi[pa-1].conj())) / (z_au[pa+1]-z_au[pa-1])).real
            j_t.append(j_r-j_l)
            
        return self.q * (simps(j_t, t_grid_au) / T_au) / T