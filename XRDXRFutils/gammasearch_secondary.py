from .spectra import FastSpectraXRD
from .gaussnewton import GaussNewton
from .gammasearch import GammaSearch, GammaMap
from numpy import (array, full, zeros, nanargmin, nanargmax, newaxis, append,
    concatenate, sqrt, average, square, std, asarray, unravel_index, minimum, where)
from numpy.linalg import pinv
from multiprocessing import Pool, cpu_count
from functools import partial
from joblib import Parallel, delayed
from platform import system



class GammaSearch_Secondary(GammaSearch):
    """
    Searches for secondary phases and compares them to primary phases.
    To be created with gammasearch_1 that contains already fitted primary phases.
    """
    def __init__(self, gammasearch_1, phases, spectrum, sigma = 0.2, **kwargs):
        super().__init__(phases, spectrum, sigma, **kwargs)
        self.gammasearch_1 = gammasearch_1
        self.set_opt(gammasearch_1.opt.copy(), copy = True)


    def overlap_area_difference(self):
        overlap_primary = self.gammasearch_1.overlap_total()[newaxis, ...]
        overlaps_secondary = self.overlap()
        overlaps_difference = overlaps_secondary - overlap_primary
        overlaps_difference = where(overlaps_difference < 0, 0, overlaps_difference)
        return overlaps_difference.sum(axis = 1)


    def overlap_area_difference_ratio(self):
        intensity_corrected = where(self.intensity < 0, 0, self.intensity)
        integral_intensity = intensity_corrected.sum()
        return self.overlap_area_difference() / integral_intensity


class GammaMap_Secondary(GammaMap):

    def from_data(self, gammamap_1, data, phases, sigma = 0.2, **kwargs):

        self.phases = phases
        self.shape = (data.shape[0], data.shape[1], -1)

        d = data.shape[0] * data.shape[1]
        spectra = [FastSpectraXRD().from_Dataf(data, i) for i in range(d)]
        self += [GammaSearch_Secondary(gs1, phases, spectrum, sigma, **kwargs) for gs1, spectrum in zip(gammamap_1, spectra)]

        return self


    def fit_cycle(self, verbose = True, **kwargs):
        x = GammaMap_Secondary(self.fit_cycle_core(verbose, **kwargs))
        x.phases = self.phases
        x.shape = self.shape
        return x


    def overlap_area_difference(self):
        return array([gs.overlap_area_difference() for gs in self]).reshape(self.shape)

    def overlap_area_difference_ratio(self):
        return array([gs.overlap_area_difference_ratio() for gs in self]).reshape(self.shape)
