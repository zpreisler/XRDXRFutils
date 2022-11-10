from .database import Phase, PhaseList
from .data import DataXRD
from .spectra import SpectraXRD, FastSpectraXRD
from .gaussnewton import GaussNewton
from numpy import (ndarray, array, full, zeros, ones, nan, isnan, nanargmin, nanargmax, newaxis,
    append, concatenate, sqrt, average, square, std, asarray, unravel_index, ravel_multi_index,
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
    Searches for phases against the given XRD experimental pattern.
    The basic structure is a list of GaussNewton objects, one for each phase.
    """

    def __init__(self, phases, spectrum, merge_peaks = True, **kwargs):
        super().__init__([GaussNewton(phase, spectrum, merge_peaks, **kwargs) for phase in phases])
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


    def select(self, phase_selected, method = None):
        if phase_selected is None:
            self.idx = self.phase_presence(method = method, correction = False).argmax()
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


    def search(self, phase_selected = None, method = None, alpha = 1):
        self.fit_cycle(steps = 2, gamma = True, alpha = alpha, downsample = 3)
        self.fit_cycle(steps = 6, a = True, s = True, gamma = True, alpha = alpha, downsample = 3)
        self.fit_cycle(steps = 2, a = True, s = True, gamma = True, alpha = alpha, downsample = 2)

        self.select(phase_selected, method)
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

    def adjustment_ratio(self, downsample = None):
        return array([gn.adjustment_ratio(downsample) for gn in self])

    def phase_presence(self, downsample = None, method = None, correction = None):
        return array([gn.phase_presence(downsample, method, correction) for gn in self])

    def L1loss(self, downsample = None):
        return array([gn.L1loss(downsample) for gn in self])

    def MSEloss(self, downsample = None):
        return array([gn.MSEloss(downsample) for gn in self])

    def metrics(self, downsample = None, method = None, correction = None):
        return self.L1loss(downsample), self.MSEloss(downsample), self.phase_presence(downsample, method, correction)


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
    """
    Map that searches for phases in every pixel of the given data.
    The basic structure is a list of GammaSearch objects, one for each pixel.
    """

    ### Creation ###

    def __init__(self, list_gammasearch = []):
        super().__init__(list_gammasearch)
        self.attribute_names_to_set = ['phases', 'indices_sel', 'n_pixels', 'shape', 'coordinates']


    def from_data(self, data, phases, indices_sel = None, merge_peaks = True, **kwargs):
        """
        Builds the map that searches for given phases in given XRD data.
        
        Arguments
        ---------
        - data: (DataXRD)
            Contains the experimental XRD patterns for every pixel.
        - phases: (list of Phase)
            Phases that will be fitted to experimental XRD patterns.
        - indices_sel: (numpy array)
            2d numpy array of boolean type, of the same dimensions as data, telling for each pixel if it is included or not in the map.
            The default value is None, in which case all the pixels are included.
        - merge_peaks: (bool)
            Whether or not to merge peaks with angular distance less than sigma.
        - kwargs: (different types, optional)
            Arguments that will be passed down to Phase.get_theta().
            They put restrictions on which peaks of tabulated phases are chosen to build synthetic XRD patterns.
        """
        if indices_sel is None:
            indices_sel = ones(data.shape[:2], bool)

        if data.shape[:2] != indices_sel.shape:
            raise Exception('Method from_data: incompatible shapes of data and indices_sel.')

        self.phases = phases
        self.indices_sel = indices_sel
        self.n_pixels = indices_sel.sum()
        self.shape = (data.shape[0], data.shape[1], len(phases), data.shape[2])

        self.coordinates = []
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                if indices_sel[y, x]:
                    self.coordinates.append((x, y))
                    spectrum = FastSpectraXRD().from_Data(data, x, y)
                    self += [GammaSearch(phases, spectrum, merge_peaks, **kwargs)]

        return self


    @property
    def type_of_elements(self):
        if len(self) > 0:
            return type(self[0])
        else:
            raise Exception('GammaMap: cannot identify the type of elements because the map is empty.')


    ### Manipulation ###

    def set_attributes_from(self, map):
        for attr_name in self.attribute_names_to_set:
            if hasattr(map, attr_name):
                setattr(self, attr_name, getattr(map, attr_name))
        return self


    def copy(self):   # CAUTION: this method performs shallow copy
        map = type(self)([gs for gs in self])
        map.set_attributes_from(self)
        return map


    def downsample(self, level):
        for gs in self:
            gs.downsample(level)
        return self


    def get_x_y(self, i):
        return self.coordinates[i]

    def get_index(self, x, y):
        if (x, y) in self.coordinates:
            return self.coordinates.index((x, y))
        else:
            raise Exception(f'Pixel of coordinates {x, y} is not in the map.')

    def get_pixel(self, x, y):
        return self[self.get_index(x, y)]


    ### Calculations for fit ###

    @staticmethod
    def f_to_parallelize(gs, f, kwargs):
        return f(gs, **kwargs)

    def parallelized(self, verbose, f, **kwargs):
        """
        Calculates function f in every element (GammaSearch) of the map.

        Arguments
        ---------
        - verbose: (bool)
            Whether to print the number of CPUs that are used for calculations.
        - f: (method of GammaSearch)
            Method that will be calculated in parallel for each element of the map.
        - kwargs: (different types, optional)
            Arguments that will be passed to f().

        Return
        ------
        List of the values returned by f(**kwargs) applied to each element of the map.
        """
        if system() == 'Darwin':
            n_cpu = cpu_count()
            if verbose:
                print(f'Using {n_cpu} CPUs')
            return Parallel(n_jobs = n_cpu)( delayed(self.f_to_parallelize)(gs, f, kwargs) for gs in self )
        else:
            n_cpu = cpu_count() - 2
            if verbose:
                print(f'Using {n_cpu} CPUs')
            with Pool(n_cpu) as p:
                return p.map(partial(self.f_to_parallelize, f = f, kwargs = kwargs), self)


    def fit(self, **kwargs):
        for gs in self:
            gs.fit(**kwargs)
        return self

    def fit_cycle(self, verbose = True, **kwargs):
        list_result = self.parallelized(verbose, self.type_of_elements.fit_cycle, **kwargs)
        map = type(self)(list_result)
        map.set_attributes_from(self)
        return map

    def search(self, verbose = True, phase_selected = None, method = None, alpha = 1):
        list_result = self.parallelized(verbose, self.type_of_elements.search, phase_selected = phase_selected, method = method, alpha = alpha)
        map = type(self)(list_result)
        map.set_attributes_from(self)
        return map


    ### Output results ###

    def format_as_1d_from_2d(self, x):
        if type(x) != ndarray:
            raise Exception('format_as_1d_from_2d requires a ndarray as parameter.')
        cols, rows = zip(*self.coordinates)
        return x[rows, cols]

    def format_as_2d_from_1d(self, x):
        if type(x) != ndarray:
            raise Exception('format_as_2d_from_1d requires a ndarray as parameter.')
        shape = list(self.shape[:2]) + list(x.shape[1:])
        x_formatted = full(shape, nan, float)
        cols, rows = zip(*self.coordinates)
        x_formatted[rows, cols] = x
        return x_formatted


    def opt(self):
        return self.format_as_2d_from_1d(array([gs.opt for gs in self]))

    def z(self):
        return self.format_as_2d_from_1d(array([gs.z() for gs in self]))

    def z0(self):
        return self.format_as_2d_from_1d(array([gs.z0() for gs in self]))

    def area(self):
        return self.format_as_2d_from_1d(array([gs.area() for gs in self]))

    def area0(self):
        return self.format_as_2d_from_1d(array([gs.area0() for gs in self]))

    def overlap_area(self, verbose = True, downsample = None):
        list_result = self.parallelized(verbose, self.type_of_elements.overlap_area, downsample = downsample)
        return self.format_as_2d_from_1d(asarray(list_result))

    def overlap_area_ratio(self, verbose = True, downsample = None):
        list_result = self.parallelized(verbose, self.type_of_elements.overlap_area_ratio, downsample = downsample)
        return self.format_as_2d_from_1d(asarray(list_result))

    def adjustment_ratio(self, verbose = True, downsample = None):
        list_result = self.parallelized(verbose, self.type_of_elements.adjustment_ratio, downsample = downsample)
        return self.format_as_2d_from_1d(asarray(list_result))

    def phase_presence(self, verbose = True, downsample = None, method = None, correction = None):
        list_result = self.parallelized(verbose, self.type_of_elements.phase_presence, downsample = downsample, method = method, correction = correction)
        return self.format_as_2d_from_1d(asarray(list_result))

    def L1loss(self, verbose = True, downsample = None):
        list_result = self.parallelized(verbose, self.type_of_elements.L1loss, downsample = downsample)
        return self.format_as_2d_from_1d(asarray(list_result))

    def MSEloss(self, verbose = True, downsample = None):
        list_result = self.parallelized(verbose, self.type_of_elements.MSEloss, downsample = downsample)
        return self.format_as_2d_from_1d(asarray(list_result))

    def metrics(self, verbose = True, downsample = None, method = None, correction = None):
        list_result = self.parallelized(verbose, self.type_of_elements.metrics, downsample = downsample, method = method, correction = correction)
        m = self.format_as_2d_from_1d(asarray(list_result))
        return (m[:, :, i, :] for i in range(m.shape[2]))

    def selected(self):
        return self.format_as_2d_from_1d(array([gs.idx for gs in self]))


    ### Misc ###

    def select_phases(self, criterion, offset = -8):
        phases_new = []

        for i_phase in range(len(self.phases)):
            criterion_sel_flat = criterion[..., i_phase][self.indices_sel]                # Criterion in selected pixels
            indices_sorted = criterion_sel_flat.argsort()                                 # Indices sorted according to 'criterion'
            indices_sorted_clean = indices_sorted[: (~isnan(criterion_sel_flat)).sum()]   # Remove indices corresponding to nan values of 'criterion'
            if not (-len(indices_sorted_clean) <= offset < len(indices_sorted_clean)):
                raise Exception(f'{self.phases[i_phase].label}: {len(indices_sorted_clean)} pixels with valid criterion. Chosen offset {offset} is out of range.')
            i_pixel = indices_sorted_clean[offset]

            gauss_newton = self[i_pixel][i_phase]
            phase_made = gauss_newton.make_phase()
            phase_made.set_name('created_%d'%i_phase)
            phase_made.set_point(i_pixel)
            phases_new += [phase_made]

        return phases_new
