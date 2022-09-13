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
        for gn in self:
            if copy:
                gn.opt = self.opt.copy()
            else:
                gn.opt = self.opt


    def select(self, phase_selected):
        if phase_selected is None:
            self.idx = self.overlap_area(downsample = 3).argmax()
        else:
            self.idx = phase_selected
        self.selected = self[self.idx]


    def fit(self, **kwargs):
        for gn in self:
            gn.fit(**kwargs)
        return self


    def fit_cycle(self, **kwargs):
        for gn in self:
            gn.fit_cycle(**kwargs)
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


    def z(self):
        return array([gn.z() for gn in self])

    def z0(self):
        return array([gn.z0() for gn in self])

    def area(self):
        return array([gn.area() for gn in self])

    def area0(self):
        return array([gn.area0() for gn in self])

    def overlap(self, downsample = None):
        return array([gn.overlap(downsample) for gn in self])

    def overlap_area(self, downsample = None):
        return array([gn.overlap_area(downsample) for gn in self])

    def overlap_area_ratio(self, downsample = None):
        return array([gn.overlap_area_ratio(downsample) for gn in self])

    def L1loss(self):
        return array([gn.L1loss() for gn in self])

    def MSEloss(self):
        return array([gn.MSEloss() for gn in self])

    def metrics(self, downsample = None):
        return self.L1loss(), self.MSEloss(), self.overlap_area(downsample)


    def overlap_total(self):
        arr_z = array([gn.z() for gn in self])
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


class GammaMap_Base(list):

    def fit(self, **kwargs):
        for gs in self:
            gs.fit(**kwargs)
        return self


    @staticmethod
    def fit_cycle_service(x, kwargs):
        return x.fit_cycle(**kwargs)

    def fit_cycle_core(self, verbose, **kwargs):
        if system() == 'Darwin':
            n_cpu = cpu_count()
            if verbose:
                print(f'Using {n_cpu} CPUs')
            result = Parallel(n_jobs = n_cpu)( delayed(gs.fit_cycle)(**kwargs) for gs in self )
        else:
            n_cpu = cpu_count() - 2
            if verbose:
                print(f'Using {n_cpu} CPUs')
            with Pool(n_cpu) as p:
                result = p.map(partial(self.fit_cycle_service, kwargs = kwargs), self)
        return result


    @staticmethod
    def search_service(x, phase_selected, alpha):
        return x.search(phase_selected = phase_selected, alpha = alpha)

    def search_core(self, phase_selected, alpha, verbose):
        if system() == 'Darwin':
            n_cpu = cpu_count()
            if verbose:
                print(f'Using {n_cpu} CPUs')
            result = Parallel(n_jobs = n_cpu)( delayed(gs.search)(phase_selected = phase_selected, alpha = alpha) for gs in self )
        else:
            n_cpu = cpu_count() - 2
            if verbose:
                print(f'Using {n_cpu} CPUs')
            with Pool(n_cpu) as p:
                result = p.map(partial(self.search_service, phase_selected = phase_selected, alpha = alpha), self)
        return result


    @staticmethod
    def metrics_service(x, downsample):
        return x.metrics(downsample = downsample)

    def metrics(self, downsample = None, verbose = True):
        if system() == 'Darwin':
            n_cpu = cpu_count()
            if verbose:
                print(f'Using {n_cpu} CPUs')
            results = Parallel(n_jobs = n_cpu)( delayed(gs.metrics)(downsample = downsample) for gs in self )
        else:
            n_cpu = cpu_count() - 2
            if verbose:
                print(f'Using {n_cpu} CPUs')
            with Pool(n_cpu) as p:
                results = p.map(partial(self.metrics_service, downsample = downsample), self)

        results = asarray(results)
        L1loss = results[:,0,:].reshape(self.shape)
        MSEloss = results[:,1,:].reshape(self.shape)
        overlap_area = results[:,2,:].reshape(self.shape)
        return L1loss, MSEloss, overlap_area


    @staticmethod
    def overlap_area_ratio_service(x, downsample):
        return x.overlap_area_ratio(downsample = downsample)

    def overlap_area_ratio(self, downsample = None, verbose = True):
        if system() == 'Darwin':
            n_cpu = cpu_count()
            if verbose:
                print(f'Using {n_cpu} CPUs')
            results = Parallel(n_jobs = n_cpu)( delayed(gs.overlap_area_ratio)(downsample = downsample) for gs in self )
        else:
            n_cpu = cpu_count() - 2
            if verbose:
                print(f'Using {n_cpu} CPUs')
            with Pool(n_cpu) as p:
                results = p.map(partial(self.overlap_area_ratio_service, downsample = downsample), self)
        return asarray(results).reshape(self.shape)


    def opt(self):
        return array([gs.opt for gs in self]).reshape(self.shape)

    def z(self):
        return array([gs.z() for gs in self]).reshape([self.shape[i] for i in range(len(self.shape) - 1)] + [len(self.phases), -1])

    def z0(self):
        return array([gs.z0() for gs in self]).reshape([self.shape[i] for i in range(len(self.shape) - 1)] + [len(self.phases), -1])

    def area(self):
        return array([gs.area() for gs in self]).reshape(self.shape)

    def area0(self):
        return array([gs.area0() for gs in self]).reshape(self.shape)

    def overlap_area(self, downsample = None):
        return array([gs.overlap_area(downsample = downsample) for gs in self]).reshape(self.shape)

    def L1loss(self):
        return array([gs.L1loss() for gs in self]).reshape(self.shape)

    def MSEloss(self):
        return array([gs.MSEloss() for gs in self]).reshape(self.shape)

    def selected(self):
        return array([gs.idx for gs in self]).reshape([self.shape[i] for i in range(len(self.shape) - 1)])


class GammaMap_Partial(GammaMap_Base):

    def from_data(self, data, phases, indices_sel, sigma = 0.2, **kwargs):

        self.phases = phases
        self.shape = (indices_sel.sum(), -1)
        self.coordinates = []

        spectra = []
        for x in range(data.shape[1]):
            for y in range(data.shape[0]):
                if indices_sel[y, x]:
                    spectra.append(FastSpectraXRD().from_Data(data, x, y))
                    self.coordinates.append((x, y))
        self += [GammaSearch(phases, spectrum, sigma, **kwargs) for spectrum in spectra]

        return self


    def fit_cycle(self, verbose = True, **kwargs):
        x = GammaMap_Partial(self.fit_cycle_core(verbose, **kwargs))
        x.phases = self.phases
        x.shape = self.shape
        x.coordinates = self.coordinates
        return x


    def search(self, phase_selected = None, alpha = 1, verbose = True):
        x = GammaMap_Partial(self.search_core(phase_selected = phase_selected, alpha = alpha, verbose = verbose))
        x.phases = self.phases
        x.shape = self.shape
        x.coordinates = self.coordinates
        return x


    def get_x_y(self, i):
        return self.coordinates[i]

    def get_index(self, x, y):
        return self.coordinates.index((x, y))

    def get_pixel(self, x, y):
        return self[self.get_index(x, y)]


class GammaMap(GammaMap_Base):
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


    def fit_cycle(self, verbose = True, **kwargs):
        x = GammaMap(self.fit_cycle_core(verbose, **kwargs))
        x.phases = self.phases
        x.shape = self.shape
        return x


    def search(self, phase_selected = None, alpha = 1, verbose = True):
        x = GammaMap(self.search_core(phase_selected = phase_selected, alpha = alpha, verbose = verbose))
        x.phases = self.phases
        x.shape = self.shape
        return x


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
