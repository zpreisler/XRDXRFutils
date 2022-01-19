from .spectra import SpectraXRD
class GaussNewton(SpectraXRD):
    """
    Class to calculate Gauss-Newton minimization of the synthetic and the experimental spectra.
    """
    def __init__(self,phase,spectra,max_theta = 53, min_intensity = 0.05):
        """
        phase: tabulated phase; Phase or PhaseList class
        spectra: experimental spectra; Spectra class
        """
        super().__init__()

        self.max_theta = max_theta
        self.min_intensity = min_intensity

        self.phase = phase
        self.spectra = spectra

        """
        Spectra
        """
        self.opt = spectra.opt.copy()
        self.channel = spectra.channel
        self.intensity = spectra.intensity

        """
        Phases

        tabulated theta: mu
        tabulated intensity: I
        """
        self.mu,self.I = self.get_theta(max_theta = max_theta,min_intensity = min_intensity)

    def get_theta(self,**kwargs):
        return self.phase.get_theta(**kwargs)
