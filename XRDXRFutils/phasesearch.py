from .spectra import SpectraXRD
from .gaussnewton import GaussNewton
from numpy import array, trapz
#from multiprocessing import Pool
from joblib import Parallel, delayed


class PhaseSearch(list):
    """
    Class to perform phase search. One experimental spectrum vs multiple phases, all with the same calibration.
    """
    def __init__(self, phases, spectrum, **kwargs):
        super().__init__([GaussNewton(phase, spectrum, **kwargs) for phase in phases])
        self.spectrum = spectrum
        self.intensity = spectrum.intensity
        self.opt = self[0].opt
        for g in self:
            g.opt = self.opt

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


    def select(self):
        self.idx = self.overlap_area().argmax()
        self.selected = self[self.idx]
        return self.selected

    def fit_cycle(self, **kwargs):
        for fit_phase in self:
            fit_phase.fit_cycle(**kwargs)
        return self

    def search(self, max_steps = 4, alpha = 1):
        self.fit_cycle(max_steps = max_steps, gamma = True, alpha = alpha)
        self.select().fit_cycle(max_steps = max_steps, a = True, s = True, gamma = True, alpha = alpha)
        self.fit_cycle(max_steps = max_steps, gamma = True, alpha = alpha)
        return self


class PhaseMap():
    def __init__(self, data, phases, **kwargs):
        self.shape_data = data.shape
        self.phases = phases
        self.phases.get_theta(**kwargs)
        self.opt_initial = data.opt
        self.list_phase_search = Parallel(n_jobs = -1)(
            delayed(self.gen_phase_search)(x) for x in data.data.reshape(-1, self.shape_data[2])
        )

    def gen_phase_search(self, x, **kwargs):
        return PhaseSearch(
            self.phases,
            SpectraXRD().from_array(x).calibrate_from_parameters(self.opt_initial),
            **kwargs
        )

    def search(self):
        self.list_phase_search = Parallel(n_jobs = -1)(
            delayed(ps.search)() for ps in self.list_phase_search
        )
        return self

    def fit_cycle(self, **kwargs):
        self.list_phase_search = Parallel(n_jobs = -1)(
            delayed(ps.fit_cycle)(**kwargs) for ps in self.list_phase_search
        )
        return self


    def opt(self):
        return array([ps.opt for ps in self.list_phase_search]).reshape((self.shape_data[0], self.shape_data[1], -1))

    def map_best_index(self):
        return array([ps.idx for ps in self.list_phase_search]).reshape(self.shape_data[0:2])

    def map_intensity(self):
        return array([trapz(ps.intensity) for ps in self.list_phase_search]).reshape(self.shape_data[0:2])

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
