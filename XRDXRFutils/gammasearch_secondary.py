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
    def __init__(self, gammasearch_1, phases, merge_peaks = True, **kwargs):
        super().__init__(phases, gammasearch_1.spectrum, merge_peaks, **kwargs)
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
    """
    Map that searches for secondary phases in every pixel of the given primary map.
    The basic structure is a list of GammaSearch_Secondary objects, one for each pixel.
    """

    def __init__(self, list_gammasearch = []):
        super().__init__(list_gammasearch)
        self.attribute_names_to_set += ['primary_phases']


    def from_data(self, gammamap_1, phases, merge_peaks = True, **kwargs):
        """
        Builds the map that searches for given secondary phases, comparing them with phases in the given primary map.

        Arguments
        ---------
        - gammamap_1: (GammaMap)
            Instance of GammaMap, here acting as primary map with its phases already fitted to data.
        - phases: (list of Phase)
            Secondary phases that will be compared to primary phases.
        - merge_peaks: (bool)
            Whether or not to merge peaks with angular distance less than sigma.
        - kwargs: (different types, optional)
            Arguments that will be passed down to Phase.get_theta().
            They put restrictions on which peaks of tabulated phases are chosen to build synthetic XRD patterns.
        """
        self.set_attributes_from(gammamap_1)
        self.primary_phases = gammamap_1.phases
        self.phases = phases
        self += [GammaSearch_Secondary(gs_1, phases, merge_peaks, **kwargs) for gs_1 in gammamap_1]
        return self


    def overlap_area_difference(self):
        return self.format_as_2d_from_1d(array([gs.overlap_area_difference() for gs in self]))

    def overlap_area_difference_ratio(self):
        return self.format_as_2d_from_1d(array([gs.overlap_area_difference_ratio() for gs in self]))
