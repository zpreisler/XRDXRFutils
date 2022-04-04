from .database import Phase, PhaseList
from .data import DataXRD
from .spectra import SpectraXRD,FastSpectraXRD
from .gaussnewton import GaussNewton
from numpy import array, full, zeros, nanargmin, nanargmax, newaxis, append, concatenate, sqrt, average, square, std
from numpy.linalg import pinv
from multiprocessing import Pool
from joblib import Parallel, delayed
import os
import pickle
import pathlib

class ChiSearch(list):
    """
    Iterate gamma.
    """
    def __init__(self, phases, spectrum, sigma = 0.2, **kwargs):

        super().__init__([GaussNewton(phase, spectrum, sigma = sigma, **kwargs) for phase in phases])

        self.opt = self[0].opt.copy()
        for gaussnewton in self:
            gaussnewton.opt = self.opt

        self.spectrum = spectrum
        self.intensity = spectrum.intensity

        chi = 1 / len(self)
        self.g = full((1, len(self)), self.iw(chi))

    def overlap_area(self):
        return array([gauss_newton.overlap_area() for gauss_newton in self])

    """
    Redefined variables
    """
    @staticmethod
    def iw(x):
        return (4 * x**2 -1) / (4 * x)

    @staticmethod
    def w(x):
        return 0.5 * (sqrt(x**2 + 1) + x)

    @staticmethod
    def der_w(x):
        return 0.5 * (x / sqrt(x**2 + 1) + 1)

    @property
    def chi(self):
        return self.w(self.g)

    def calculate_components(self):

        self.z_phases = concatenate([gauss_newton.z()[:, newaxis] for gauss_newton in self], axis = 1)
        self.z_components = self.chi * self.z_phases

    def del_components(self):

        del self.z_phases
        del self.z_components

    def z_decomposed(self):

        z_phases = concatenate([gauss_newton.z()[:, newaxis] for gauss_newton in self], axis = 1)
        z_components = self.chi * z_phases

        return z_components

    def z(self):

        z_phases = concatenate([gauss_newton.z()[:, newaxis] for gauss_newton in self], axis = 1)
        z_components = self.chi * z_phases

        value = self.z_components.sum(axis = 1)

        return value

    def der_f_a_s_beta(self):

        der = tuple([zeros((self.intensity.shape[0], 1))]) * 3

        for gauss_newton, chi in zip(self, self.chi.squeeze()):

            gauss_newton.calculate_components()

            der = tuple( x + y for x, y in zip( der, (chi * d for d in gauss_newton.der_f_a_s_beta())))

            gauss_newton.del_components()

        return der


    def der_f_chi(self):

        return self.der_w(self.g) * self.z_phases
        
    def fit_phases(self, a = False, s = False, beta = False, chi = True, alpha = 1):

        self.calculate_components()

        n_opt = a + s + beta
        n_chi = chi * len(self)

        Jacobian_construction = []

        # Calibration parameters
        if (n_opt > 0):
            der_f_a, der_f_s, der_f_beta = self.der_f_a_s_beta()
            if a:
                Jacobian_construction.append(der_f_a)
            if s:
                Jacobian_construction.append(der_f_s)
            if beta:
                Jacobian_construction.append(der_f_beta)

        # Chi
        if chi:
            Jacobian_construction.append(self.der_f_chi())

        # Jacobian
        Jacobian_f = concatenate(Jacobian_construction, axis = 1)
            
        # Evolution of parameters

        y = self.intensity[:, newaxis]
        f = self.z_components.sum(axis = 1, keepdims = True)
        r = y - f

        try:
            evol = pinv(Jacobian_f) @ r # or scipy.linalg.pinv
        except:
            evol = full((Jacobian_f.shape[1], 1), 0)

        d_params = alpha * evol
        mask_opt = [a, s, beta]
        self.opt[mask_opt] += d_params[0:n_opt, 0]

        if chi:
            self.g += d_params[n_opt : (n_opt + n_chi)].T

        self.del_components()

        return self


    def fit_cycle(self, steps = 8, **kwargs):

        for i in range(steps):
            self.fit_phases(**kwargs)

        return self


    def search(self, alpha = 1):

        self.fit_cycle(3, chi = True, alpha = alpha)
        self.fit_cycle(6, a = True, s = True, alpha = alpha)
        self.fit_cycle(3, chi = True, alpha = alpha)

        return self

class ChiMap(list):
    """
    Construct gamma phase maps.
    """
    def from_data(self,data,phases,sigma = 0.2, **kwargs):
        
        self.phases = phases

        d = data.shape[0] * data.shape[1]
        spectra = [FastSpectraXRD().fromDataf(data,i) for i in range(d)]

        self += [ChiSearch(phases,spectrum,sigma) for spectrum in spectra]

        return self

    @staticmethod
    def f_search(x):
        return x.search()

    def search(self):
        with Pool(50) as p:
            result = p.map(self.f_search, self)
        return ChiMap(result)
