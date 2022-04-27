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

class GammaSearch_Secondary(GammaSearch):
    """
    Searches for secondary phases and compares them to primary phases.
    To be created with gammasearch_1 that contains already fitted primary phases.
    """
    def __init__(self, gammasearch_1, phases, spectrum, sigma = 0.2, **kwargs):
        super().__init__(phases, spectrum, sigma, **kwargs)

        self.gammasearch_1 = gammasearch_1

        self.opt = gammasearch_1.opt.copy()
        for gaussnewton in self:
            gaussnewton.opt = self.opt.copy()


    # def overlap_ratio(self):
        