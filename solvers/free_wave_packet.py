#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module allows to compare Pseudo-Espectral, Runge-Kutta, and
Crank-Nicolson methods of time evolution of a free wave packet
according to the Schroedinger Equation.

/AU/i stands for Atomic Units
/SI/i stands for International System of Units
"""

# libraries
import time
import numpy as np
import pandas as pd
from scipy.fftpack import fft, ifft, fftfreq
from scipy.integrate import simps
from scipy.sparse import diags
from scipy.linalg import inv

from .generic_potential import GenericPotential

class FreeWavePacket(GenericPotential):
    """
    This class allows the simulation of a free wave packet propagating
    in one direction
    """

    def __init__(self, L=100, N=1024):
        """
        Parameters
        ----------
        L : float
            the system's size in Angstrom
        N : int
            the number of points
        """
        super(FreeWavePacket,self).__init__(N=N)
        self.L_nm = L * 10
        self.z_nm = np.linspace(-self.L_nm/2, self.L_nm/2, self.N)

    def analytical_evolution(self, zi=-20.0, zf=20, E=150.0, \
        deltaz=5.0, L=250.0, N=8192):
        """
        Evolves a free wave packet with energy `E` and dispersion 
        `deltaz` from a initial position `zi` until a final position 
        `zf`. The evolution takes place in a space of length `L` split 
        in `N` points. This is a pseudo-analytical solution, since it 
        assumes that a numerical integration might be done with great 
        precision.

        Parameters
        ----------
        zi : float
            initial position in Angstrom
        zf : float 
            final position in Angstrom
        E : float
            wave energy in eV
        deltaz : float
            initial dispersion in Angstrom
        L : float
            the length of the space in Angstrom
        N : integer
            the number of points in space
        
        Return
        -------
        summary : dict
            The summary keys are
            - `L` the length of the space
            - `N` the number of points in space
            - `z_si` the z grid in SI
            - `z_nm` the z grid in nm
            - `z_ang` the z grid in Angstrom
            - `z_au` the z grid in AU
            - `z_ang` the z grid in Angstrom
            - `wave_initial_au` the initial wave packet in AU
            - `wave_final_au` the final wave packet in AU
            - `a_initial` initial <psi|psi>
            - `a_final` final <psi|psi>
            - `conservation` 100 * a_final / a_initial
            - `stdev` the final standard deviation in Angstrom
            - `skewness` the final skewness
            - `time` the time for the wave depart from zi and arrive 
                at zf
            - `zf_real` the real final position, which might differ 
                from zf
        """
        assert zf > zi # wave should go from left to right
        assert E > 0 # zero energy, wave stays in place
        assert L > 0 # space length must not be null
        assert int(np.log2(N)) == np.log2(N) # must be a power of 2
        assert -L/4 < zf < L/4 # final position might cause errors

        # change to AU
        L_au = L / self.au2ang
        E_au = E / self.au2ev
        deltaz_au = deltaz / self.au2ang
        zi_au = zi / self.au2ang
        zf_au = zf / self.au2ang
        k0_au = np.sqrt(2 * E_au)

        # direct and reciproval meshes
        z_au = np.linspace(-L_au/2.0, L_au/2.0, N)
        dz_au = np.abs(z_au[1] - z_au[0])
        k_au = fftfreq(N, d=dz_au)

        # times
        time_aux = 1e-18
        time = 5e-18
        
        # initial values
        zm_au = zi_au
        zm_au_aux = zi_au
        
        # initial wave packet
        PN = 1 / (2 * np.pi * deltaz_au ** 2) ** (1 / 4)
        psi = PN * \
            np.exp(1j*k0_au*z_au-(z_au-zi_au)**2/(4*deltaz_au**2))
        psi_initial = np.copy(psi) # salva uma copia

        # initial values
        A = A0 = np.sqrt(simps(np.abs(psi) ** 2, z_au))
        zm_au = zi_au
        stdev_au = deltaz_au
        skewness = 0.0
        
        while np.abs(zm_au - zf_au) >= 0.00001:
            # new time
            t_au = (time) / self.au_t
            
            # initial wave packet
            psi = np.copy(psi_initial)

            # acctual pseudo-analytic solotion
            psi_k = fft(psi)
            omega_k = k_au**2 / 2
            psi = ifft(psi_k * np.exp(-1j * omega_k * t_au))

            # main indicators
            A2 = simps(np.abs(psi)**2, z_au).real
            A = np.sqrt(A2)
            psic = np.conjugate(psi)
            zm_au = (simps(psic * z_au * psi, z_au)).real / A2
            
            # adjust time step
            if np.abs(zm_au - zf_au) >= 0.00001:
                if zm_au_aux < zf_au < zm_au or \
                    zm_au < zf_au < zm_au_aux:
                    aux = (time_aux-time) / 2
                elif zf_au < zm_au and zf_au < zm_au_aux:
                    aux = - abs(time_aux-time)
                elif zf_au > zm_au and zf_au > zm_au_aux:
                    aux = abs(time_aux-time)
                    
                time_aux = time
                time += aux
                zm_au_aux = zm_au
                
                continue
            
            # secondary indicators
            zm2 = simps(psic * z_au ** 2 * psi, z_au).real / A2
            zm3 = simps(psic * z_au ** 3 * psi, z_au).real / A2
            stdev_au = np.sqrt(np.abs(zm2-zm_au**2))
            skewness = (zm3-3*zm_au*stdev_au**2-zm_au**3)/stdev_au**3
        
        return {
            'L': L,
            'N': N,
            'z_si': z_au * self.au2ang * 1e-10,
            'z_nm': z_au * self.au2ang * 0.1,
            'z_ang': z_au * self.au2ang,
            'z_au': z_au,
            'wave_initial': psi_initial,
            'wave_final': psi,
            'a_initial': A0,
            'a_final': A,
            'conservation': 100 * A / A0,
            'stdev': stdev_au * self.au2ang,
            'skewness': skewness,
            'time': time,
            'zf_real': zm_au * self.au2ang,
        }

    def numerical_evolution(self, zi=-20.0, zf=20, E=150.0, \
        deltaz=5.0, method='pe', L=100.0, N=512, dt=1e-19):
        """
        Evolves a free wave packet with energy `E` from an initial 
        position `zi` until a final position `zf` using the given 
        `method`. The evolution takes place in a space of length `L`
        split in `N` points. Each time step has size `dt`

        Parameters
        ----------
        zi : float
            the wave's initial position in Angstrom
        zf : float
            the wave's final position in Angstrom
        deltaz : float
            the wave's initial spread or stardard deviation
        E : float
            the wave's energy in eV
        method : string
            the method, the possibilities are:
            - 'pe' for Pseudo-Espectral
            - 'cn' for Crank-Nicolson
            - 'rk' for Runge-Kutta
        L : float
            the length of the space in Angstrom
        N : integer
            the number of points in space
        dt : float
            the time step in seconds

        Return
        -------
        summary : dict
            The summary keys are
            - `L` the length of the space
            - `N` the number of points in space
            - `dt` the time step
            - `method` the method used for evolution
            - `z_au` grid space in Angstrom
            - `z_nm` the z grid in nm
            - `z_ang` the z grid in Angstrom
            - `z_si` grid space in SI
            - `wave_initial` the initial wave packet
            - `wave_final` the final wave packet
            - `a_initial` the initial <psi|psi>
            - `a_final` the final <psi|psi>
            - `conservation` 100 * a_final / a_initial
            - `stdev` the final standard deviation in Angstrom
            - `skewness` the final skewness
            - `zf_real` the real final position, which might differ 
                from zf
            - `time_total` the time the program takes
            - `iterations` the numer of iterations the program makes
        """
        if not method in ['pe', 'rk', 'cn']:
            raise Exception('Invalid method [%s]' % method)
        
        assert zf > zi # wave should go from left to right
        assert E > 0 # zero energy, wave stays in place
        assert L > 0 # space length must not be null
        assert int(np.log2(N)) == np.log2(N) # must be a power of 2
        assert -L/4 < zf < L/4 # final position might cause errors

        # device settings
        self.N = N
        self.L_nm = L / 10.0
        self.z_nm = np.linspace(-self.L_nm/2, self.L_nm/2, self.N)
        self.v_ev = np.zeros(N, dtype=np.complex_)
        self.m_eff = np.ones(N, dtype=np.complex_)

        # change to AU
        E_au = E / self.au2ev
        deltaz_au = deltaz / self.au2ang
        zi_au = zi / self.au2ang
        zf_au = zf / self.au2ang
        k0_au = np.sqrt(2 * E_au)

        # start measuring time here because some matrices are very time
        # consuming
        time_inicial = time.time()

        # normalize device
        self.normalize_device(method=method, reset=True)
        z_au = self.device.z_au

        # the evolution operator takes Psi(z,t) into Psi(z,t+dt)
        evolution_operator = lambda p: self.evolve_real(p,0)
    
        # initial wave packet
        PN = 1 / (2 * np.pi * deltaz_au ** 2) ** (1 / 4)
        psi = PN*np.exp(1j*k0_au*z_au-(z_au-zi_au)**2/(4*deltaz_au**2))
        psi_initial = np.copy(psi) # stores a copy

        # initial <psi|psi>
        A = A0 = np.sqrt(simps(np.abs(psi) ** 2, z_au))

        # initial values
        zm_au = zi_au
        stdev_au = deltaz_au
        skewness = 0.0
        iterations = 0

        while zm_au < zf_au:
            psi = evolution_operator(psi)
            iterations += 1

            # if the wave walks to the worng side, inverst the wave 
            # vector it is expect to happen one time in the most, and 
            # in the beginning
            if zm_au < zi_au:
                k0_au *= -1.0
                psi = np.copy(psi_initial) * np.exp(-1.0)
                zm_au = zi_au
                continue

            # main indicators
            A2 = simps(np.abs(psi) ** 2, z_au).real
            A = np.sqrt(A2)
            psic = np.conjugate(psi)

            # average position of psi
            zm_au = (simps(psic * z_au * psi, z_au)).real / A2

            # stop measuring time (it might be the end)
            time_final = time.time()
            
            # if the final position has been achieved of the time
            # is greater than 1000 seconds
            if zm_au >= zf_au or time_final - time_inicial > 1000:
                # secondary indicators
                zm2 = simps(psic * z_au ** 2 * psi, z_au).real / A2
                zm3 = simps(psic * z_au ** 3 * psi, z_au).real / A2
                stdev_au = np.sqrt(np.abs(zm2-zm_au**2))
                skewness = (zm3-3*zm_au*stdev_au**2-zm_au**3)/stdev_au**3
                
                if time_final - time_inicial > 1000:
                    break

        program_total_time = time_final - time_inicial
        
        return {
            'L': L,
            'N': N,
            'dt': dt,
            'method': method,
            'z_ang': z_au * self.au2ang,
            'z_si': z_au * self.au2ang * 1e-10,
            'z_nm': z_au * self.au2ang * 0.1,
            'z_au': z_au,
            'wave_initial': psi_initial,
            'wave_final': psi,
            'a_initial': A0,
            'a_final': A,
            'conservation': 100 * A / A0,
            'stdev': stdev_au * self.au2ang,
            'skewness': skewness,
            'time_total': program_total_time,
            'iterations': iterations,
            'zf_real': zm_au * self.au2ang
        }