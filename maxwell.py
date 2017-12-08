#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 08:25:49 2017, completed Nov 20 2017

@author: johninglesfield
"""
import numpy as np
from cmath import exp
from numpy import sqrt, sin, cos, tan, pi, trace
from scipy.linalg import eig, solve
from scipy.special import jv, jvp
from math import atan2

"""
The Waveguide class contains the functions for waveguide applications of
electromagnetic embedding. The class is initiated with the value of dielectric
constant, the wave-vector along the waveguide (which has width 2*pi), and D
which defines the basis functions.
"""

class Waveguide:
    def __init__(self, epsilon, ky):
        """
        Width of waveguide d = 2*pi, dielectric of walls = epsilon
        wavevector in y-direction down waveguide = ky
        distance defining basis functions in units of d = Dbar
        """
        self.d = 2.0*pi
        self.epsilon = epsilon
        self.ky = ky
        return
    
    def match(self, w):
        kx = sqrt(w**2 - self.ky**2)
        gam = sqrt(self.ky**2 - self.epsilon*w**2)
        f = [tan(kx*pi), -1.0/tan(kx*pi), gam/(self.epsilon*kx)]
        return f
        
    def matrix_setup_t(self, M, D_tilde):
        """
        Matrix elements of transverse waves only, size of basis  = (2*M + 1).
        D_tilde defines basis functions, with D_tilde = D/d.
        A1 = first part of A matrix (curl of basis functions)
        A2 = matrix elements multiplying embedding potential
        B1 = overlap matrix
        """
        fac = 1.0/D_tilde
        self.A1 = np.zeros((2*M+1, 2*M+1))
        self.A2 = np.zeros((2*M+1, 2*M+1))
        self.B1 = np.zeros((2*M+1, 2*M+1))
        for m in range(-M, M+1):
            gm = fac*m
            for n in range(-M, M+1):
                gn = fac*n
                if m == n:
                    dint = self.d
                else:
                    dint = 2.0*sin(0.5*(gn - gm)*self.d)/(gn - gm)
                self.A1[m, n] = (self.ky*self.ky + gm*gm)*(self.ky*self.ky + 
                       gn*gn)*dint
                self.A2[m, n] = 2.0*gm*gn*cos(0.5*(gn - gm)*self.d)
                self.B1[m, n] = (self.ky*self.ky + gm*gn)*dint
        return
    
    def matrix_eigen(self, w0):
        """
        Evaluates eigenvalues of embedded matrix, embedding "potential"
        evaluated at trial frequency w0. Returns ordered frequencies.
        """
        gamma = sqrt(self.ky*self.ky - self.epsilon*w0*w0)
        sigma = self.epsilon*w0*w0/gamma
        dsigma = (self.epsilon/gamma)*(1.0 + 
                 0.5*self.epsilon*w0*w0/(gamma*gamma))
        A = self.A1 - self.A2*(sigma - w0*w0*dsigma)
        B = self.B1 + self.A2*dsigma
        eigenvalues = sorted(eig(A, B)[0].real)
        return sqrt([abs(ww) for ww in eigenvalues])         
    
    def spectral_density(self, w, eta):
        """
        Evaluates electric spectral density with E-field basis
        at complex frequency (w, eta).
        """
        ww = complex(w*w, eta)
        kx = sqrt(self.epsilon*ww - self.ky*self.ky)
        sigma = 1j*self.epsilon*ww/kx
        green = self.A1 - sigma*self.A2 - ww*self.B1
        Gamma = 2.0*w*trace(solve(green, self.B1)).imag/pi
        return Gamma

    def matrix_setup_tl(self, M, N, D_tilde):
        """
        Matrix elements of transverse and longitudinal E-field basis, 
        size of basis = (2*M + 1) transverse + (2*N + 1) longitudinal. 
        """
        fac = 1.0/D_tilde        
        self.A1 = np.zeros((2*(M+N)+2, 2*(M+N)+2))
        self.A2 = np.zeros((2*(M+N)+2, 2*(M+N)+2))
        self.B1 = np.zeros((2*(M+N)+2, 2*(M+N)+2))
        """
        Evaluates tranverse-transverse matrix elements
        """
        i = -1
        for m in range(-M, M+1):
            i = i + 1
            gm = fac*m
            j = -1
            for n in range(-M, M+1):
                j = j + 1
                gn = fac*n
                if m == n:
                    dint = self.d
                else:
                    dint = 2.0*sin(0.5*(gn - gm)*self.d)/(gn - gm)
                self.A1[i, j] = (self.ky*self.ky + gm*gm)*(self.ky*self.ky + 
                       gn*gn)*dint
                self.A2[i, j] = 2.0*gm*gn*cos(0.5*(gn - gm)*self.d)
                self.B1[i, j] = (self.ky*self.ky + gm*gn)*dint
        """
        Evaluates longitudinal-longitudinal matrix elements
        """
        for m in range(-N, N+1):
            i = i + 1
            gm = fac*m
            j = 2*M
            for n in range(-N, N+1):
                j = j + 1
                gn = fac*n
                if m == n:
                    dint = self.d
                else:
                    dint = 2.0*sin(0.5*(gn - gm)*self.d)/(gn - gm)
                self.A2[i, j] = 2.0*self.ky*self.ky*cos(0.5*(gn - gm)*self.d)
                self.B1[i, j] = (self.ky*self.ky + gm*gn)*dint
        """
        Evaluates transverse-longitudinal matrix elements
        """
        i = -1
        for m in range(-M, M+1):
            i = i + 1
            gm = fac*m
            j = 2*M
            for n in range(-N, N+1):
                j = j + 1
                gn = fac*n
                if m == n:
                    dint = self.d
                else:
                    dint = 2.0*sin(0.5*(gn - gm)*self.d)/(gn - gm)
                self.A2[i, j] = -2.0*gm*self.ky*cos(0.5*(gn - gm)*self.d)
                self.B1[i, j] = self.ky*(gn - gm)*dint
                self.A2[j, i] = self.A2[i, j]
                self.B1[j, i] = self.B1[i, j]
        return
        
    def matrix_setup_lexact(self, M, D_tilde):
        """
        Matrix elements of transverse + exact Laplace.
        """
        fac = 1.0/D_tilde
        self.A1 = np.zeros((2*M+3, 2*M+3), dtype=np.complex)
        self.A2 = np.zeros((2*M+3, 2*M+3), dtype=np.complex)
        self.B1 = np.zeros((2*M+3, 2*M+3), dtype=np.complex)
        """
        Evaluates tranverse-transverse matrix elements
        """
        i = -1
        for m in range(-M, M+1):
            i = i + 1
            gm = fac*m
            j = -1
            for n in range(-M, M+1):
                j = j + 1
                gn = fac*n
                if m == n:
                    dint = self.d
                else:
                    dint = 2.0*sin(0.5*(gn - gm)*self.d)/(gn - gm)
                self.A1[i, j] = (self.ky*self.ky + gm*gm)*(self.ky*self.ky + 
                       gn*gn)*dint
                self.A2[i, j] = 2.0*gm*gn*cos(0.5*(gn - gm)*self.d)
                self.B1[i, j] = (self.ky*self.ky + gm*gn)*dint
        """
        Evaluates Laplace-Laplace matrix elements
        """
        i = 2*M + 1
        self.A2[i, i] = (exp(self.ky*self.d)+exp(-self.ky*self.d))*self.ky**2
        self.A2[i+1, i+1] = self.A2[i, i]
        self.B1[i, i] = (exp(self.ky*self.d)-exp(-self.ky*self.d))*self.ky
        self.B1[i+1, i+1] = self.B1[i, i]
        self.A2[i, i+1] = 2.0*self.ky**2
        self.A2[i+1, i] = 2.0*self.ky**2
        """
        Evaluates transverse-Laplace matrix elements
        """
        i = -1
        j = 2*M + 1
        for m in range(-M, M+1):
            i = i + 1
            gm = fac*m
            self.A2[i, j] = -1j*self.ky*gm*(exp(0.5*(self.ky-1j*gm)*self.d) + 
                   exp(-0.5*(self.ky-1j*gm)*self.d))
            self.A2[i, j+1] = 1j*self.ky*gm*(exp(0.5*(self.ky+1j*gm)*self.d) + 
                   exp(-0.5*(self.ky+1j*gm)*self.d))
            self.B1[i, j] = self.ky*(exp(0.5*(self.ky-1j*gm)*self.d) - 
                   exp(-0.5*(self.ky-1j*gm)*self.d))
            self.B1[i, j+1]= self.ky*(exp(0.5*(self.ky+1j*gm)*self.d) - 
                   exp(-0.5*(self.ky+1j*gm)*self.d))
            self.A2[j, i] = self.A2[i, j].conjugate()
            self.A2[j+1, i] = self.A2[i, j+1].conjugate()
            self.B1[j, i] = self.B1[i, j].conjugate()
            self.B1[j+1, i] = self.B1[i, j+1].conjugate()
        return
    
    def matrix_setup_magnetic(self, M, D_tilde):
        """
        Matrix elements of H-field basis, size = (2*M + 1)
        A1 = first part of A matrix (curl of basis functions)
        A2 = matrix elements multiplying embedding potential
        B1 = overlap matrix
        """
        fac = 1.0/D_tilde
        self.A1 = np.zeros((2*M+1, 2*M+1))
        self.A2 = np.zeros((2*M+1, 2*M+1))
        self.B1 = np.zeros((2*M+1, 2*M+1))
        for m in range(-M, M+1):
            gm = fac*m
            for n in range(-M, M+1):
                gn = fac*n
                if m == n:
                    dint = self.d
                else:
                    dint = 2.0*sin(0.5*(gn - gm)*self.d)/(gn - gm)
                self.A1[m, n] = (gm*gn + self.ky**2)*dint
                self.A2[m, n] = 2.0*cos(0.5*(gn - gm)*self.d)
                self.B1[m, n] = dint
        return
    
    def spectral_density_magnetic(self, w, eta):
        """
        Spectral density with H-basis at complex frequency = (w, eta)
        Gamma_el = electric spectral density
        Gamma_mag = magnetic spectral density
        """
        ww = complex(w*w, eta)
        kx = sqrt(self.epsilon*ww - self.ky*self.ky)
        sigma = 1j*kx/self.epsilon
        green = self.A1 - sigma*self.A2 - ww*self.B1
        Gamma_el = 2.0*w*(trace(solve(green, self.A1))/ww).imag/pi
#        Gamma_mag = 2.0*w*trace(solve(green, self.B1)).imag/pi
        return Gamma_el

"""
The Cylinder class contains the functions used in calculating the density of
states in a two-dimensional square lattice of metallic cylinders, with the 
Drude dielectric function. The polarization is such that the magnetic field 
lies along the cylinder axis, taken as the z-direction. The class in initiated
by specifying the radius of the cylinder, the number of m-values used to 
describe the fields (m is the axzimuthal quantum number), the plasmon
frequency, and the plasmon lifetime. 
"""
    
class Cylinder:
    def __init__(self, rho_tilde, nm, wp, tau):
        """
        Lattice constant = 2*pi, 
        radius of cylinders relative to lattice constant = rho_tilde,
        number of m-values = nm,
        Drude dielectric constant calculated with
        plasmon frequency = wp, plasmon lifetime = tau.
        """
        self.d = 2.0*pi
        self.r = rho_tilde*self.d
        self.nm = nm
        self.A = self.d**2
        self.wp = wp
        self.tau = tau
        return
    
    def recips(self, kx, ky, M):
        """
        Sets up reciprocal lattice, number of recips = M,
        Wave-vector = (kx, ky).
        Outputs k[i, 0] = gx + kx, k[i, 1] = gx + ky, k[i, 2] = |k + g|.
        """
        self.M = M
        g = []
        i = -1
        for m in range(-20, 21):
            for n in range(-20, 21):
                i = i+1
                g.append((m, n, m*m+n*n))
        g = sorted(g, key=lambda x:x[2])
        self.k = np.zeros((M, 3))
        self.phi = np.zeros(M)
        for i in range(M):
            self.k[i, 0] = g[i][0] + kx
            self.k[i, 1] = g[i][1] + ky
            self.k[i, 2] = sqrt(self.k[i, 0]**2 + self.k[i, 1]**2)
            self.phi[i] = atan2(self.k[i, 1], self.k[i, 0])
        return
    
    def matrix(self):
        """
        Matrix elements with H-field basis
        A1 = first part of A matrix (curl of basis functions)
        A2 = matrix elements multiplying embedding potential
        B1 = overlap matrix
        """
        self.A1 = np.zeros((self.M, self.M))
        self.A2 = np.zeros((self.M, self.M, self.nm))
        self.B1 = np.zeros((self.M, self.M))
        for i in range(self.M):
            for j in range(self.M):
                if i==j:
                    self.B1[i, j] = self.A - pi*self.r**2
                else:
                    kk = (sqrt((self.k[i, 0]-self.k[j,0])**2  
                        + (self.k[i, 1]-self.k[j,1])**2))
                    self.B1[i, j] = -2.0*pi*self.r*jv(1, kk*self.r)/kk
                k_k = self.k[i, 0]*self.k[j, 0] + self.k[i, 1]*self.k[j, 1]
                self.A1[i, j] = k_k*self.B1[i, j]
                self.A2[i, j, 0] = (jv(0, self.k[i, 2]*self.r)
                                  *jv(0, self.k[j, 2]*self.r))
                for m in range(self.nm):
                    self.A2[i, j, m] = (2.0*jv(m, self.k[i, 2]*self.r)
                                    *jv(m, self.k[j, 2]*self.r) 
                                    *cos(m*(self.phi[i]-self.phi[j])))
        return
    
    def spectral_density(self, w):
        """
        Spectral density withj H-basis at real frequency w
        gamma = electric spectral density
        """
        epsilon = 1.0 - self.wp**2/(w*(w + 1.0j/self.tau))
        kappa = w*sqrt(epsilon)
        fac = np.zeros(self.nm, dtype=np.complex)
        for m in range(self.nm):
            fac[m] = jvp(m, kappa*self.r)/jv(m, kappa*self.r)
        fac = 2.0*pi*self.r*kappa*fac/epsilon
        green = self.A1 + np.dot(self.A2, fac) - w*w*self.B1
        gamma = 2.0*(trace(solve(green, self.A1))).imag/(w*pi)
        return gamma           
                    
        
        
                    
                
                    
                    
               
                
                
                
                
    
    


                
        
    
            
        
        
        
        
        
        
        
        
                
        
            
        
        
