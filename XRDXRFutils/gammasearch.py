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

import gc

class GammaSearch(list):
    """
    Iterate gamma.
    """
    def __init__(self, phases, spectrum, sigma = 0.2, **kwargs):

        super().__init__([GaussNewton(phase, spectrum, sigma = sigma, **kwargs) for phase in phases])

        self.spectrum = spectrum
        self.intensity = spectrum.intensity

        self.opt = self[0].opt.copy()
        for gaussnewton in self:
            gaussnewton.opt = self.opt

    def select(self):

        #self.idx = (self.overlap3_area() * self.spectrum.rescaling).argmax()
        self.idx = self.overlap3_area().argmax()
        self.selected = self[self.idx]

        return self.selected

    def fit_cycle(self, **kwargs):
        for gauss_newton in self:
            gauss_newton.fit_cycle(**kwargs)

        return self

    def search(self, alpha = 1):

        self.fit_cycle(steps = 4, gamma = True, alpha = alpha, downsample = 3)

        selected = self.select()
        selected.fit_cycle(steps = 2, a = True, s = True, gamma = True, alpha = alpha, downsample = 3)
        selected.fit_cycle(steps = 2, a = True, s = True, gamma = True, alpha = alpha, downsample = 2)
        selected.fit_cycle(steps = 2, a = True, s = True, gamma = True, alpha = alpha)

        self.fit_cycle(steps = 1, gamma = True, alpha = alpha,downsample = 3)
        self.fit_cycle(steps = 1, gamma = True, alpha = alpha,downsample = 2)
        self.fit_cycle(steps = 2, gamma = True, alpha = alpha)

        return self

    def search_kb(self, alpha = 1):

        k,b = self.kb

        self.fit_cycle(steps = 3, gamma = True, alpha = alpha)

        selected = self.select()
        selected.fit_cycle(steps = 6,
                k = k, b = b,
                gamma = True, alpha = alpha)

        self.fit_cycle(steps = 4, gamma = True, alpha = alpha)

        return self

    def area(self):
        return array([gauss_newton.area() for gauss_newton in self])

    def area0(self):
        return array([gauss_newton.area0() for gauss_newton in self])

    def overlap_area(self):
        return array([gauss_newton.overlap_area() for gauss_newton in self])

    def L1loss(self):
        return array([gauss_newton.L1loss() for gauss_newton in self])

    def MSEloss(self):
        return array([gauss_newton.MSEloss() for gauss_newton in self])

    def overlap3_area(self):
        return array([gauss_newton.overlap3_area() for gauss_newton in self])

class GammaMap(list):
    """
    Construct gamma phase maps.
    """
    def from_data(self,data,phases,sigma = 0.2, **kwargs):
        
        self.phases = phases
        self.shape = (data.shape[0] , data.shape[1], -1)

        d = data.shape[0] * data.shape[1]
        spectra = [FastSpectraXRD().fromDataf(data,i) for i in range(d)]

        self += [GammaSearch(phases,spectrum,sigma) for spectrum in spectra]

        return self

    @staticmethod
    def f_search(x):
        return x.search()

    def search(self):
        with Pool(50) as p:
            result = p.map(self.f_search, self)
        x = GammaMap(result)

        x.phases = self.phases
        x.shape = self.shape

        return x

    def opt(self):
        return array([phase_search.opt for phase_search in self]).reshape(self.shape)

    def area(self):
        return array([phase_search.area() for phase_search in self]).reshape(self.shape)

    def area0(self):
        return array([phase_search.area0() for phase_search in self]).reshape(self.shape)

    def overlap_area(self):
        return array([phase_search.overlap_area() for phase_search in self]).reshape(self.shape)

    def overlap3_area(self):
        return array([phase_search.overlap3_area() for phase_search in self]).reshape(self.shape)

    def L1loss(self):
        return array([phase_search.L1loss() for phase_search in self]).reshape(self.shape)

    def MSEloss(self):
        return array([phase_search.MSEloss() for phase_search in self]).reshape(self.shape)

    def selected(self):
        return array([phase_search.idx for phase_search in self]).reshape(self.shape)

    def get_index(self,x,y):
        return x + y * self.shape[1]

    def get_pixel(self,x,y):
        return self[x + y * self.shape[1]]
