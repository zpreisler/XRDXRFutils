from .data import DataXRD
from .spectra import SpectraXRD
from .gaussnewton import GaussNewton
from numpy import array
#from multiprocessing import Pool
from joblib import Parallel, delayed
import os
import pickle


PHASE_SEARCH__N_JOBS = -2


class PhaseSearch(list):
    """
    Class to perform phase search. One experimental spectrum vs multiple phases, all with the same calibration.
    """
    def __init__(self, phases, spectrum, **kwargs):
        super().__init__([GaussNewton(phase, spectrum, **kwargs) for phase in phases])
        self.spectrum = spectrum
        self.intensity = spectrum.intensity
        self.set_opt(self[0].opt)
        self.k_b = None


    ### Misc ###
    def set_relation_a_s(self, tuple_k_b):
        self.k_b = tuple_k_b
        return self

    def set_opt(self, opt):
        self.opt = opt.copy()
        for g in self:
            g.opt = self.opt


    ### Fit ###
    def select(self):
        self.idx = self.overlap_area().argmax()
        self.selected = self[self.idx]
        return self.selected

    def fit_cycle(self, **kwargs):
        for fit_phase in self:
            fit_phase.fit_cycle(**kwargs)
        return self

    def search(self, max_steps = (4, 8, 4), alpha = 1):
        self.fit_cycle(max_steps = max_steps[0], gamma = True, alpha = alpha)
        if self.k_b is None:
            self.select().fit_cycle(max_steps = max_steps[1], a = True, s = True, gamma = True, alpha = alpha)
        else:
            self.select().fit_cycle(max_steps = max_steps[1], k = self.k_b[0], b = self.k_b[1], gamma = True, alpha = alpha)
        self.fit_cycle(max_steps = max_steps[2], gamma = True, alpha = alpha)
        return self


    ### Output ###
    def loss(self):
        return array([g.loss() for g in self])

    def fit_error(self):
        return array([g.fit_error() for g in self])

    def area_fit(self):
        return array([g.area_fit() for g in self])

    def area_0(self):
        return array([g.area_0() for g in self])

    def area_min_0_fit(self):
        return array([g.area_min_0_fit() for g in self])

    def overlap_area(self):
        return array([g.overlap_area() for g in self])
    
    def component_ratio(self):
        return array([g.component_ratio() for g in self])


class PhaseMap():
    ### Initialization ###
    def __init__(self, data, phases, **kwargs):
        self.shape_data = data.shape
        self.phases = phases
        self.phases.get_theta(**kwargs)
        self.opt_initial = data.opt
        self.k_b = None
        self.list_phase_search = Parallel(n_jobs = PHASE_SEARCH__N_JOBS)(
            delayed(self.gen_phase_search)(x, **kwargs) for x in data.data.reshape(-1, self.shape_data[2])
        )

    def gen_phase_search(self, x, **kwargs):
        return PhaseSearch(
            self.phases,
            SpectraXRD().from_array(x).calibrate_from_parameters(self.opt_initial),
            **kwargs
        )


    ### Misc ###
    def set_relation_a_s(self, tuple_k_b):
        self.k_b = tuple_k_b
        for ps in self.list_phase_search:
            ps.set_relation_a_s(tuple_k_b)
        return self

    def get_pixel(self, x, y):
        return self.list_phase_search[y * self.shape_data[1] + x]


    ### Fit ###
    def search(self, **kwargs):
        self.list_phase_search = Parallel(n_jobs = PHASE_SEARCH__N_JOBS)(
            delayed(ps.search)(**kwargs) for ps in self.list_phase_search
        )
        return self

    def fit_cycle(self, **kwargs):
        self.list_phase_search = Parallel(n_jobs = PHASE_SEARCH__N_JOBS)(
            delayed(ps.fit_cycle)(**kwargs) for ps in self.list_phase_search
        )
        return self


    ### Output ###
    def opt(self):
        return array([ps.opt for ps in self.list_phase_search]).reshape((self.shape_data[0], self.shape_data[1], -1))

    def map_best_index(self):
        return array([ps.idx for ps in self.list_phase_search]).reshape(self.shape_data[0:2])

    def map_intensity(self):
        return array([ps.intensity.sum() for ps in self.list_phase_search]).reshape(self.shape_data[0:2])

    def map_counts(self):
        return array([ps.spectrum.counts.sum() for ps in self.list_phase_search]).reshape(self.shape_data[0:2])

    def map_counts_clean(self):
        return array([ps.spectrum.counts_clean.sum() for ps in self.list_phase_search]).reshape(self.shape_data[0:2])

    def loss(self):
        return array([ps.loss() for ps in self.list_phase_search]).reshape((self.shape_data[0], self.shape_data[1], -1))

    def fit_error(self):
        return array([ps.fit_error() for ps in self.list_phase_search]).reshape((self.shape_data[0], self.shape_data[1], -1))

    def area_fit(self):
        return array([ps.area_fit() for ps in self.list_phase_search]).reshape((self.shape_data[0], self.shape_data[1], -1))

    def area_0(self):
        return array([ps.area_0() for ps in self.list_phase_search]).reshape((self.shape_data[0], self.shape_data[1], -1))

    def overlap_area(self):
        return array([ps.overlap_area() for ps in self.list_phase_search]).reshape((self.shape_data[0], self.shape_data[1], -1))

    def component_ratio(self):
        return array([ps.component_ratio() for ps in self.list_phase_search]).reshape((self.shape_data[0], self.shape_data[1], -1))
    
    def component_ratio_2(self):
        return Parallel(n_jobs = PHASE_SEARCH__N_JOBS)(
            delayed(ps.component_ratio)() for ps in self.list_phase_search
        )


class PhaseMapSave():
    def __init__(self, phasemap):
        self.opt_initial = phasemap.opt_initial
        self.phases = phasemap.phases
        self.k_b = phasemap.k_b
        self.list_opt = [ps.opt for ps in phasemap.list_phase_search]
        self.list_g = [[gn.g for gn in ps] for ps in phasemap.list_phase_search]
        self.list_tau = [[gn.tau for gn in ps] for ps in phasemap.list_phase_search]
        self.min_theta = phasemap.list_phase_search[0][0].min_theta
        self.max_theta = phasemap.list_phase_search[0][0].max_theta
        self.min_intensity = phasemap.list_phase_search[0][0].min_intensity
        self.first_n_peaks = phasemap.list_phase_search[0][0].first_n_peaks

    def reconstruct_phase_map(self, path_xrd):
        if os.path.isfile(path_xrd + 'xrd.h5'):
            data = DataXRD().load_h5(path_xrd + 'xrd.h5')
        else:
            data = DataXRD().read_params(path_xrd + 'Scanning_Parameters.txt').read(path_xrd)
        data.calibrate_from_parameters(self.opt_initial)

        pm = PhaseMap(data, self.phases, min_theta = self.min_theta, max_theta = self.max_theta, min_intensity = self.min_intensity, first_n_peaks = self.first_n_peaks)
        if self.k_b is not None:
            pm.set_relation_a_s(self.k_b)
        for i in range(len(self.list_opt)):
            ps = pm.list_phase_search[i]
            ps.set_opt(self.list_opt[i])
            for j in range(len(ps)):
                gn = ps[j]
                gn.g = self.list_g[i][j]
                gn.tau = self.list_tau[i][j]
        return pm

    def save_to_file(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_from_file(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)