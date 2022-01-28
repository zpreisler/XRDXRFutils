from .spectra import SpectraXRD

from numpy import exp,log,pi,array,ones,zeros,full,trapz,minimum,fabs,sign,sqrt
from numpy.linalg import pinv,inv

from matplotlib.pyplot import plot

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

        self.label = phase.label

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

    def plot_spectra(self,*args,**kwargs):
        super().plot(*args,**kwargs)

    def plot(self,*args,**kwargs):
        plot(self.theta,self.z(),*args,**kwargs)

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

    @staticmethod
    def fgamma(gamma,delta=1.0):
        return 0.5 * (sqrt(1 + gamma**2)) + 0.5 * gamma
        #return log(exp(gamma - 0.5) + 1.0)
        #return delta * (sqrt(1 + (gamma/delta)**2) -1)

    @staticmethod
    def dgamma(gamma,delta=1.0):
        return 0.5 * (gamma / sqrt(1 + gamma**2)) + 0.5
        #return 1.0 / (1.0 + exp(-gamma + 0.5))
        #return gamma / (delta * sqrt(gamma**2/delta**2 + 1))

    def z(self):     
        """
        Synthetic spectra.
        """
        x = self.theta
        y = zeros(len(x))
        for mu,I,sigma2,gamma in zip(self.mu, self.I,
                                     self.sigma2, self.gamma):
            c = self.core(x,mu,sigma2)
            y += self.fgamma(gamma) * I * c
            
        return y

    def z0(self):     
        """
        Synthetic spectra.
        """
        x = self.theta
        y = zeros(len(x))
        for mu,I,sigma2,gamma in zip(self.mu, self.I,
                                     self.sigma2, self.gamma):
            c = self.core(x,mu,sigma2)
            y += I * c
            
        return y

    def loss(self):
        y = self.intensity
        return sum((y - self.z())**2)

    def area_fit(self):
        return trapz(self.z())

    def area_0(self):
        return trapz(self.z0())

    def overlap(self):
        m =  minimum(self.z(),self.intensity)
        m[m < 0] = 0
        return m

    def overlap_area(self):
        return trapz(self.overlap())

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
            #dgamma += [I * c]
            #z += gamma * I * c

            dgamma += [I * c * self.dgamma(gamma)]
            z += self.fgamma(gamma) * I * c
        
        dz = y - z
        J = array(dgamma).T
        dr = pinv(J) @ dz
   
        self.gamma[:] += dr * alpha

    def minimize_sigma(self,alpha = 1):
        """
        Minimize sigma2
        """
        x = self.theta
        y = self.intensity
        z = zeros(len(x))
        
        dsigma2 = []
        for mu,I,sigma2,gamma in zip(self.mu,self.I,
                                     self.sigma2,self.gamma):
            c = self.core(x,mu,sigma2)
            h = self.fgamma(gamma) * I * c
            dsigma2 += [h * self.dsigma2(x,mu,sigma2)]
            z += h
        
        dz = y - z
        J = array(dsigma2).T
        dr = pinv(J) @ dz
   
        self.sigma2[:] += dr * alpha

    def autocalibration(self,alpha = 1):
        """
        Autocalibration
        """

        x = self.theta
        y = self.intensity

        da = zeros(len(x))
        ds = zeros(len(x))
        dbeta = zeros(len(x))

        z = zeros(len(x))
        
        for mu,I,sigma2,gamma in zip(self.mu,self.I,
                                     self.sigma2,self.gamma):
            c = self.core(x,mu,sigma2)
            #h = gamma * I * c

            h = self.fgamma(gamma) * I * c
            
            da += h * self.da(self.channel,x,self.opt[0],self.opt[1],mu,sigma2)
            ds += h * self.ds(self.channel,x,self.opt[0],self.opt[1],mu,sigma2)
            dbeta += h * self.dbeta(x,mu,sigma2)

            z += h
        
        dz = y - z
        J = array([da,ds,dbeta]).T
        
        dr = pinv(J) @ dz
        self.opt[:] += dr * alpha

    def calibration(self,alpha = 1):
        """
        Calibrate a,s and intensities
            assuming fixed beta, and sigma
        """
        x = self.theta
        y = self.intensity

        da = zeros(len(x))
        ds = zeros(len(x))
        dbeta = zeros(len(x))

        z = zeros(len(x))
        
        dgamma = []
        for mu,I,sigma2,gamma in zip(self.mu,self.I,
                                     self.sigma2,self.gamma):
            c = self.core(x,mu,sigma2)
            #h = gamma * I * c
            h = self.fgamma(gamma) * I * c

            #dgamma += [I * c]
            dgamma += [I * c * self.dgamma(gamma)]

            da += h * self.da(self.channel,x,self.opt[0],self.opt[1],mu,sigma2)
            ds += h * self.ds(self.channel,x,self.opt[0],self.opt[1],mu,sigma2)

            z += h
        
        dz = y - z
        J = array([da,ds] + dgamma).T
        dr = pinv(J) @ dz
   
        self.opt[:2] += dr[:2] * alpha
        self.gamma[:] += dr[2:] * alpha
