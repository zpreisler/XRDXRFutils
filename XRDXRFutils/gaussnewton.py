from .spectra import SpectraXRD

from numpy import exp,pi,array,ones,zeros,full
from numpy.linalg import pinv,inv

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

        """
        parameters sigma^2, gamma
        """
        self.sigma2 = full(len(self.mu),0.04)
        self.gamma = ones(len(self.I))

    def get_theta(self,**kwargs):
        return self.phase.get_theta(**kwargs)

    def plot(self,*args,**kwargs):
        super().plot(*args,**kwargs)
        self.phase.plot()

    """
    Derivatives
    """
    @staticmethod
    def core(x,mu,sigma2):
        return exp(-0.5 * (x - mu)**2 / sigma2)

    @staticmethod
    def dsigma2(x,mu,sigma2):
        return 0.5 * (x - mu)**2 / sigma2**2

    @staticmethod
    def da(channel,x,a,s,mu,sigma2):
        return -1.0 / sigma2 * 180 / pi * s / ((a + channel)**2 + s**2) * (x - mu)
    
    @staticmethod
    def ds(channel,x,a,s,mu,sigma2):
        return 1.0 / sigma2 * 180 / pi * (a + channel) / ((a + channel)**2 + s**2) * (x - mu)
    
    @staticmethod
    def dbeta(x,mu,sigma2):
        return -1.0 / sigma2 * (x - mu)

    def z(self):     
        """
        Synthetic spectra.
        """
        x = self.theta
        y = zeros(len(x))
        for mu,I,sigma2,gamma in zip(self.mu, self.I,
                                     self.sigma2, self.gamma):
            c = self.core(x,mu,sigma2)
            y += gamma * I * c
            
        return y

    def minimize_gamma(self,alpha = 1):
        """
        Minimize gamma
        """
        x = self.theta
        y = self.intensity
        z = zeros(len(x))
        
        dgamma = []
        for mu,I,sigma2,gamma in zip(self.mu,self.I,
                                     self.sigma2,self.gamma):
            c = self.core(x,mu,sigma2)
            dgamma += [I * c]
            z += gamma * I * c
        
        dz = y - z
        J = array(dgamma).T
        dr = pinv(J) @ dz
   
        self.gamma[:] += dr * alpha
