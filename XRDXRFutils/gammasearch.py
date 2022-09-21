from .database import Phase, PhaseList
from .data import DataXRD
from .spectra import SpectraXRD, FastSpectraXRD
from .gaussnewton import GaussNewton
from numpy import (ndarray, array, full, zeros, ones, nan, nanargmin, nanargmax, newaxis, append,
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
        self.set_opt(spectrum.opt.copy(), copy = True)


    @property
    def intensity(self):
        return self.spectrum.intensity


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


    def downsample(self, level):
        for gn in self:
            gn.downsample(level)
        return self


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

    def L1loss(self, downsample = None):
        return array([gn.L1loss(downsample) for gn in self])

    def MSEloss(self, downsample = None):
        return array([gn.MSEloss(downsample) for gn in self])

    def metrics(self, downsample = None):
        return self.L1loss(downsample), self.MSEloss(downsample), self.overlap_area(downsample)


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



class GammaMap(list):

    ### Creation ###

    def from_data(self, data, phases, indices_sel = None, sigma = 0.2, **kwargs):
        if indices_sel is None:
            indices_sel = ones(data.shape[:2], bool)

        if data.shape[:2] != indices_sel.shape:
            raise Exception('Incompatible shapes of data and indices_sel')

        self.phases = phases
        self.indices_sel = indices_sel
        self.shape = (data.shape[0], data.shape[1], len(phases), data.shape[2])

        self.coordinates = []
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                if indices_sel[y, x]:
                    spectrum = FastSpectraXRD().from_Data(data, x, y)
                    self.coordinates.append((x, y))
                    self += [GammaSearch(phases, spectrum, sigma, **kwargs)]

        return self


    @property
    def n_pixels(self):
        return self.indices_sel.sum()


    ### Manipulation ###

    def downsample(self, level):
        for gs in self:
            gs.downsample(level)
        return self


    def set_attributes_from(self, map):
        for attr_name in ['phases', 'indices_sel', 'shape', 'coordinates']:
            if hasattr(map, attr_name):
                setattr(self, attr_name, getattr(map, attr_name))


    def get_x_y(self, i):
        return self.coordinates[i]

    def get_index(self, x, y):
        return self.coordinates.index((x, y))

    def get_pixel(self, x, y):
        return self[self.get_index(x, y)]


    ### Calculations for fit ###

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


    def fit_cycle(self, verbose = True, **kwargs):
        x = GammaMap(self.fit_cycle_core(verbose, **kwargs))
        x.set_attributes_from(self)
        return x


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


    def search(self, phase_selected = None, alpha = 1, verbose = True):
        x = GammaMap(self.search_core(phase_selected = phase_selected, alpha = alpha, verbose = verbose))
        x.set_attributes_from(self)
        return x


    ### Output information ###

    @staticmethod
    def metrics_service(x, downsample):
        return x.metrics(downsample = downsample)

    def metrics_core(self, downsample, verbose):
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
        return asarray(results)


    @staticmethod
    def overlap_area_ratio_service(x, downsample):
        return x.overlap_area_ratio(downsample = downsample)

    def overlap_area_ratio_core(self, downsample, verbose):
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
        return asarray(results)

    def format_as_map(self, x):
        if type(x) != ndarray:
            raise Exception('format_as_matrix requires a ndarray as parameter')
        shape = list(self.shape[:2]) + list(x.shape[1:])
        x_formatted = full(shape, nan, float)
        cols, rows = zip(*self.coordinates)
        x_formatted[rows, cols] = x
        return x_formatted


    def opt(self):
        return self.format_as_map(array([gs.opt for gs in self]))

    def z(self):
        return self.format_as_map(array([gs.z() for gs in self]))

    def z0(self):
        return self.format_as_map(array([gs.z0() for gs in self]))

    def area(self):
        return self.format_as_map(array([gs.area() for gs in self]))

    def area0(self):
        return self.format_as_map(array([gs.area0() for gs in self]))

    def overlap_area(self, downsample = None):
        return self.format_as_map(array([gs.overlap_area(downsample) for gs in self]))

    def overlap_area_ratio(self, downsample = None, verbose = True):
        return self.format_as_map(self.overlap_area_ratio_core(downsample, verbose))

    def L1loss(self):
        return self.format_as_map(array([gs.L1loss() for gs in self]))

    def MSEloss(self):
        return self.format_as_map(array([gs.MSEloss() for gs in self]))

    def metrics(self, downsample = None, verbose = True):
        m = self.format_as_map(self.metrics_core(downsample, verbose))
        return (m[:, :, i, :] for i in range(3))

    def selected(self):
        return self.format_as_map(array([gs.idx for gs in self]))


    ### Misc ###

    def select_phases(self, criterion, offset = -8):
        phases_new = []

        for i_phase in range(len(self.phases)):
            i_pixel = criterion[..., i_phase][self.indices_sel].argsort()[offset]
            gauss_newton = self[i_pixel][i_phase]
            phase_made = gauss_newton.make_phase()
            phase_made.set_name('created_%d'%i_phase)
            phase_made.set_point(i_pixel)
            phases_new += [phase_made]

        return phases_new
