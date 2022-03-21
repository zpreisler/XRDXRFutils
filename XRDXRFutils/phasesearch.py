from .gaussnewton import GaussNewton
from numpy import array


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

    def overlap_area(self):
        return array([g.overlap_area() for g in self])

    def loss(self):
        return array([g.loss() for g in self])

    def select(self):
        self.idx = self.overlap_area().argmax()
        self.selected = self[self.idx]
        return self.selected

    def fit_cycle(self, **kwargs):
        for fit_phase in self:
            fit_phase.fit_cycle(**kwargs)

    def search(self, alpha = 1.0):
        self.fit_cycle(max_steps = 4, gamma = True, alpha = alpha)
        self.select().fit_cycle(max_steps = 4, a = True, s = True, gamma = True, alpha = alpha)
        self.fit_cycle(max_steps = 4, gamma = True, alpha = alpha)
        return self


class PhaseMap(list):
    """
    Class to process images
    """      
    def from_data(self,data,phases):
    
        phases.get_theta(max_theta=53,min_intensity=0.05)
        arr = data.data.reshape(-1,1280)

        spectras = self.gen_spectras(arr)        
        for spectra in spectras:
            spectra.opt = [-1186.6, 1960.3, 51]
        
        self += [PhaseSearch(phases,spectra) for spectra in spectras]
        
        return self
    
    @staticmethod
    def f_spectra(x):
        return SpectraXRD().from_array(x)

    def gen_spectras(self,a):
        with Pool() as p:
            spectras = p.map(self.f_spectra,a)
        return spectras
    
    @staticmethod
    def f_search(x):
        return x.search()
    
    def search(self):
        with Pool() as p:
            result = p.map(self.f_search,self)
        return PhaseMap(result)
