from .database import Phase, PhaseList
from .data import DataXRD
from .spectra import SpectraXRD,FastSpectraXRD
from .gaussnewton import GaussNewton
from numpy import (array, full, zeros, nanargmin, nanargmax, newaxis, append,
    concatenate, sqrt, average, square, std, asarray, unravel_index, ravel_multi_index,
    minimum, where)
from numpy.linalg import pinv
from multiprocessing import Pool, cpu_count
from functools import partial
from joblib import Parallel, delayed
from platform import system
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
        self.set_opt(self[0].opt, copy = True)


    def set_opt(self, opt, copy = True):
        self.opt = opt
        for gaussnewton in self:
            if copy:
                gaussnewton.opt = self.opt.copy()
            else:
                gaussnewton.opt = self.opt


    def select(self, phase_selected):
        if phase_selected is None:
            self.idx = self.overlap3_area().argmax()
        else:
            self.idx = phase_selected
        self.selected = self[self.idx]


    def fit_cycle(self, **kwargs):
        for gauss_newton in self:
            gauss_newton.fit_cycle(**kwargs)
        return self


    def search(self, phase_selected = None, alpha = 1):
        self.fit_cycle(steps = 4, gamma = True, alpha = alpha, downsample = 3)
        self.fit_cycle(steps = 1, a = True, s = True, gamma = True, alpha = alpha, downsample = 2)

        self.select(phase_selected)
        self.set_opt(self.selected.opt, copy = False)

        self.selected.fit_cycle(steps = 2, a = True, s = True, gamma = True, alpha = alpha, downsample = 3)
        self.selected.fit_cycle(steps = 2, a = True, s = True, gamma = True, alpha = alpha, downsample = 2)
        self.selected.fit_cycle(steps = 2, a = True, s = True, gamma = True, alpha = alpha)

        self.fit_cycle(steps = 1, gamma = True, alpha = alpha, downsample = 3)
        self.fit_cycle(steps = 1, gamma = True, alpha = alpha, downsample = 2)
        self.fit_cycle(steps = 2, gamma = True, alpha = alpha)

        return self


    def area(self):
        return array([gauss_newton.area() for gauss_newton in self])

    def area0(self):
        return array([gauss_newton.area0() for gauss_newton in self])

    def overlap(self):
        return array([gauss_newton.overlap() for gauss_newton in self])

    def overlap_area(self):
        return array([gauss_newton.overlap_area() for gauss_newton in self])

    def L1loss(self):
        return array([gauss_newton.L1loss() for gauss_newton in self])

    def MSEloss(self):
        return array([gauss_newton.MSEloss() for gauss_newton in self])

    def overlap3_area(self):
        return array([gauss_newton.overlap3_area() for gauss_newton in self])

    def metrics(self):
        return self.L1loss(), self.MSEloss(), self.overlap3_area()


    def overlap_total(self):
        arr_z = array([gauss_newton.z() for gauss_newton in self])
        z_max = arr_z.max(axis = 0)
        m = minimum(z_max, self.intensity)
        m = where(m < 0, 0, m)
        return m

    def overlap_total_area(self):
        return self.overlap_total().sum()

    def overlap_total_ratio(self):
        integral_intersection = self.overlap_total_area()
        intensity_corrected = where(self.intensity < 0, 0, self.intensity)
        integral_intensity = intensity_corrected.sum()
        return (integral_intersection / integral_intensity)


class GammaMap(list):
    """
    Construct gamma phase maps.
    """
    def from_data(self, data, phases, sigma = 0.2, **kwargs):

        self.phases = phases
        self.shape = (data.shape[0], data.shape[1], -1)

        d = data.shape[0] * data.shape[1]
        spectra = [FastSpectraXRD().from_Dataf(data, i) for i in range(d)]
        self += [GammaSearch(phases, spectrum, sigma, **kwargs) for spectrum in spectra]

        return self


    @staticmethod
    def fit_cycle_service(x, kwargs):
        return x.fit_cycle(**kwargs)

    def fit_cycle_core(self, **kwargs):
        if system() == 'Darwin':
            n_cpu = cpu_count()
            print(f'Using {n_cpu} CPUs')
            result = Parallel(n_jobs = n_cpu)( delayed(gs.fit_cycle)(**kwargs) for gs in self )
        else:
            n_cpu = cpu_count() - 2
            print(f'Using {n_cpu} CPUs')
            with Pool(n_cpu) as p:
                result = p.map(partial(self.fit_cycle_service, kwargs = kwargs), self)
        return result

    def fit_cycle(self, **kwargs):
        x = GammaMap(self.fit_cycle_core(**kwargs))
        x.phases = self.phases
        x.shape = self.shape
        return x


    @staticmethod
    def search_service(x, phase_selected, alpha):
        return x.search(phase_selected = phase_selected, alpha = alpha)

    def search_core(self, phase_selected, alpha):
        if system() == 'Darwin':
            n_cpu = cpu_count()
            print(f'Using {n_cpu} CPUs')
            result = Parallel(n_jobs = n_cpu)( delayed(gs.search)(phase_selected = phase_selected, alpha = alpha) for gs in self )
        else:
            n_cpu = cpu_count() - 2
            print(f'Using {n_cpu} CPUs')
            with Pool(n_cpu) as p:
                result = p.map(partial(self.search_service, phase_selected = phase_selected, alpha = alpha), self)
        return result

    def search(self, phase_selected = None, alpha = 1):
        x = GammaMap(self.search_core(phase_selected = phase_selected, alpha = alpha))
        x.phases = self.phases
        x.shape = self.shape
        return x


    @staticmethod
    def metrics_service(x):
        return x.metrics()

    def metrics(self):
        if system() == 'Darwin':
            n_cpu = cpu_count()
            print(f'Using {n_cpu} CPUs')
            results = Parallel(n_jobs = n_cpu)( delayed(gs.metrics)() for gs in self )
        else:
            n_cpu = cpu_count() - 2
            print(f'Using {n_cpu} CPUs')
            with Pool(n_cpu) as p:
                results = p.map(self.metrics_service, self)

        results = asarray(results)
        L1loss = results[:,0,:].reshape(self.shape)
        MSEloss = results[:,1,:].reshape(self.shape)
        overlap3_area = results[:,2,:].reshape(self.shape)
        return L1loss, MSEloss, overlap3_area


    def opt(self):
        return array([gs.opt for gs in self]).reshape(self.shape)

    def area(self):
        return array([gs.area() for gs in self]).reshape(self.shape)

    def area0(self):
        return array([gs.area0() for gs in self]).reshape(self.shape)

    def overlap_area(self):
        return array([gs.overlap_area() for gs in self]).reshape(self.shape)

    def overlap3_area(self):
        return array([gs.overlap3_area() for gs in self]).reshape(self.shape)

    def L1loss(self):
        return array([gs.L1loss() for gs in self]).reshape(self.shape)

    def MSEloss(self):
        return array([gs.MSEloss() for gs in self]).reshape(self.shape)

    def selected(self):
        return array([gs.idx for gs in self]).reshape((self.shape[0], self.shape[1]))

    def get_x_y(self, i):
        y, x = unravel_index(i, self.shape[:2])
        return x, y

    def get_index(self, x, y):
        return ravel_multi_index((y, x), self.shape[:2])

    def get_pixel(self, x, y):
        return self[self.get_index(x, y)]

    def select_phases(self, criterion, offset = -8):
        phases_new = []

        for idx in range(len(self.phases)):
            point = criterion[:, :, idx].flatten().argsort()[offset]
            gauss_newton = self[point][idx]
            phase_made = gauss_newton.make_phase()
            phase_made.set_name('created_%d'%idx)
            phase_made.set_point(point)
            phases_new += [phase_made]

        return phases_new
