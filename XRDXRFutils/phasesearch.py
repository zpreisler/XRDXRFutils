from .spectra import SpectraXRD
from .gaussnewton import GaussNewton
from numpy import array
from multiprocessing import Pool
from joblib import Parallel, delayed


class PhaseSearch(list):
    """
    Class to perform phase search. Multiple phases vs one experimental spectrum.
    """
    def __init__(self, phases, spectrum):
        super().__init__([GaussNewton(phase, spectrum) for phase in phases])
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

    def search(self, max_steps = 4, alpha = 1):
        self.fit_cycle(max_steps = max_steps, gamma = True, alpha = alpha)
        self.select().fit_cycle(max_steps = max_steps, a = True, s = True, gamma = True, alpha = alpha)
        self.fit_cycle(max_steps = max_steps, gamma = True, alpha = alpha)
        return self


class PhaseMap(list):
    """
    Class to process images
    """      
    def from_data(self, data, phases):
        phases.get_theta(max_theta = 53, min_intensity = 0.05)
        arr = data.data.reshape(-1, 1280)
        spectra = self.gen_spectra(arr)
        for spectrum in spectra:
            spectrum.calibrate_from_parameters(data.opt)
        self += [PhaseSearch(phases, spectrum) for spectrum in spectra]
        return self

    @staticmethod
    def f_spectrum(x):
        return SpectraXRD().from_array(x)

    def gen_spectra(self, a):
        with Pool() as p:
            spectra = p.map(self.f_spectrum, a)
        return spectra

    @staticmethod
    def f_search(x):
        return x.search()

    def search(self):
        with Pool() as p:
            result = p.map(self.f_search, self)
        return PhaseMap(result)

    def search_2(self):
        result = Parallel(n_jobs = -1)(
            delayed(self.f_search)(p) for p in self
        )
        return PhaseMap(result)