from .database import Phase, PhaseList
from .data import DataXRD
from .spectra import SpectraXRD
from .gaussnewton import GaussNewton
from numpy import array, nanargmax
#from multiprocessing import Pool
from joblib import Parallel, delayed
import os
import pickle

import gc


PHASE_SEARCH__N_JOBS = -2


class PhaseSearch(list):
    """
    Class to perform phase search. One experimental spectrum vs multiple phases, all with the same calibration.
    """
    def __init__(self, phases, spectrum, **kwargs):
        # kwargs will be chained down to Phase.get_theta()
        super().__init__([GaussNewton(phase, spectrum, **kwargs) for phase in phases])

        self.kwargs = kwargs
        self.spectrum = spectrum
        self.intensity = spectrum.intensity
        self.set_opt(self[0].opt)
        self.k_b = None

        gc.collect()


    ### Misc ###
    def set_relation_a_s(self, tuple_k_b):
        self.k_b = tuple_k_b
        return self

    def set_opt(self, opt):
        self.opt = opt.copy()
        for g in self:
            g.opt = self.opt


    ### Construction ###
    def add_phases(self, phases):
        list_to_add = [GaussNewton(phase, self.spectrum, **self.kwargs) for phase in phases]
        for gn in list_to_add:
            gn.opt = self.opt
        self += list_to_add

    def remove_phases(self, list_i):
        for i in list_i:
            self.pop(i)


    ### Fit ###
    def select(self):
        self.idx = self.overlap_area().argmax()
        self.selected = self[self.idx]
        return self.selected

    def fit_cycle(self, **kwargs):
        for fit_phase in self:
            fit_phase.fit_cycle(**kwargs)

        gc.collect()

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
        # kwargs will be chained down to Phase.get_theta()
        self.kwargs = kwargs
        self.shape_data = data.shape
        self.phases = phases
        self.phases.get_theta(**kwargs)
        self.n_phases_primary = None
        self.opt_initial = data.opt
        self.k_b = None
        self.list_phase_search = Parallel(n_jobs = PHASE_SEARCH__N_JOBS)(
            delayed(self.gen_phase_search)(x) for x in data.data.reshape(-1, self.shape_data[2])
        )

        gc.collect()

    def gen_phase_search(self, x):
        return PhaseSearch(
            self.phases,
            SpectraXRD().from_array(x).calibrate_from_parameters(self.opt_initial),
            **self.kwargs
        )


    ### Misc ###
    def set_relation_a_s(self, tuple_k_b):
        self.k_b = tuple_k_b
        for ps in self.list_phase_search:
            ps.set_relation_a_s(tuple_k_b)
        return self

    def get_pixel(self, x, y):
        return self.list_phase_search[y * self.shape_data[1] + x]


    def extract_best_phases(self):
        arr_overlap_area = array([ps.overlap_area() for ps in self.list_phase_search])
        list_phases = []

        for idx_phase in range(len(self.phases)):
            idx_point = nanargmax(arr_overlap_area[:, idx_phase])
            gn = self.list_phase_search[idx_point][idx_phase]
            mu, I = gn.get_theta(**self.kwargs)
            I_new = I * gn.gamma.squeeze()
            I_new /= I_new.max()

            phase_new = Phase(gn.phase)
            phase_new.theta = mu
            phase_new.intensity = I_new
            phase_new.label = gn.phase.label
            list_phases.append(phase_new)

        return PhaseList(list_phases)


    ### Construction ###
    def add_phases(self, phases):
        print(f'Current phases: {self.phases.label}', flush = True)
        list_labels_present = [p.label for p in self.phases]
        phases_sel = PhaseList([p for p in phases if p.label not in list_labels_present])
        print(f'Adding {phases_sel.label}...', flush = True)
        self.phases += phases_sel
        for ps in self.list_phase_search:
            ps.add_phases(phases_sel)
        print(f'New phases: {self.phases.label}')

    def remove_phases(self, labels_phase):
        list_labels_present = [p.label for p in self.phases]
        print(f'Current phases: {self.phases.label}')
        list_i = []
        for label in labels_phase:
            try:
                i = list_labels_present.index(label)
                # if i < self.n_phases_primary:
                #     print(f'{label} cannot be removed because it was used to set the calibration.')
                # else:
                list_i.append(i)
            except ValueError:
                print(f'{label} is not present among the stored phases.')
        print(f'Phases to be removed: {[self.phases[i].label for i in list_i]}')

        if list_i:
            list_i.sort(reverse = True)
            for i in list_i:
                self.phases.pop(i)
            for ps in self.list_phase_search:
                ps.remove_phases(list_i)
        print(f'New phases: {self.phases.label}')


    ### Fit ###
    def search(self, **kwargs):
        self.n_phases_primary = len(self.phases)
        self.list_phase_search = Parallel(n_jobs = PHASE_SEARCH__N_JOBS)(
            delayed(ps.search)(**kwargs) for ps in self.list_phase_search
        )

        gc.collect()

        return self

    def fit_cycle(self, **kwargs):
        self.list_phase_search = Parallel(n_jobs = PHASE_SEARCH__N_JOBS)(
            delayed(ps.fit_cycle)(**kwargs) for ps in self.list_phase_search
        )

        gc.collect()
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
    
    # def component_ratio_2(self):
    #     return Parallel(n_jobs = PHASE_SEARCH__N_JOBS)(
    #         delayed(ps.component_ratio)() for ps in self.list_phase_search
    #     )


class PhaseMapSave():
    def __init__(self, phasemap):
        self.opt_initial = phasemap.opt_initial
        self.phases = phasemap.phases
        self.n_phases_primary = phasemap.n_phases_primary
        self.k_b = phasemap.k_b
        self.list_opt = [ps.opt for ps in phasemap.list_phase_search]
        self.list_g = [[gn.g for gn in ps] for ps in phasemap.list_phase_search]
        self.list_tau = [[gn.tau for gn in ps] for ps in phasemap.list_phase_search]
        self.kwargs = phasemap.kwargs

    def reconstruct_phase_map(self, path_xrd):
        if os.path.isfile(path_xrd + 'xrd.h5'):
            data = DataXRD().load_h5(path_xrd + 'xrd.h5')
        else:
            data = DataXRD().read_params(path_xrd + 'Scanning_Parameters.txt').read(path_xrd)
        data.calibrate_from_parameters(self.opt_initial)

        pm = PhaseMap(data, self.phases, **self.kwargs)
        pm.n_phases_primary = self.n_phases_primary
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
