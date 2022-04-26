from .database import Phase, PhaseList
from .data import DataXRD
from .spectra import SpectraXRD,FastSpectraXRD
from .gaussnewton import GaussNewton
from .gammasearch import GammaSearch
from numpy import (array, full, zeros, nanargmin, nanargmax, newaxis, append,
    concatenate, sqrt, average, square, std, asarray, unravel_index, minimum, where)
from numpy.linalg import pinv
from multiprocessing import Pool, cpu_count
from functools import partial
from joblib import Parallel, delayed
from platform import system
import os
import pickle
import pathlib

import gc

class GammaSearch_Distinguish():
    """
    Searches for secondary phases and compares them to primary phases.
    To be created with gammasearch_1 that contains already fitted primary phases.
    """
    def __init__(self, gammasearch_1, phases_2, spectrum, sigma = 0.2, **kwargs):
        self.gammasearch_1 = gammasearch_1
        self.gammasearch_2 = GammaSearch(phases_2, spectrum, sigma = sigma, **kwargs)

        self.spectrum = spectrum
        self.intensity = spectrum.intensity

        self.opt = self[0].opt.copy()
        for gaussnewton in self:
            gaussnewton.opt = self.opt.copy()


    # def overlap_ratio(self):
        