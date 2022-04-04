from .spectra import SpectraXRD, FastSpectraXRD

from numpy import sum, exp, log, pi, array, ones, zeros, full, full_like, trapz, minimum, maximum, std, fabs, sign, sqrt, square, average, clip, newaxis, concatenate, append, where
from numpy.linalg import pinv, inv

from scipy.optimize import newton

from matplotlib.pyplot import plot

import gc

class GaussNewton(FastSpectraXRD):
    """
    Class to calculate Gauss-Newton minimization of the synthetic and the experimental spectrum.
    """
    def __init__(self, phase, spectrum, sigma = 0.2, **kwargs):
        # kwargs will be passed to Phase.get_theta()
        """
        phase: tabulated phase; Phase or PhaseList class
        spectrum: experimental spectrum; Spectra class
        """
        super().__init__()

        self.phase = phase
        self.spectrum = spectrum

        self.label = phase.label

        """
        Spectrum
        """

        self.opt = spectrum.opt.copy()

        # Variables along the channels
        #self.channel = spectrum.channel[:, newaxis]
        #self.intensity = spectrum.intensity[:, newaxis]

        self.channel = spectrum.channel
        self.intensity = spectrum.intensity

        self.channel = 0.5 * (self.channel[::2] + self.channel[1::2])
        self.intensity = 0.5 * (self.intensity[::2] + self.intensity[1::2])

        self.channel = 0.5 * (self.channel[::2] + self.channel[1::2])
        self.intensity = 0.5 * (self.intensity[::2] + self.intensity[1::2])

        """
        Phases

        tabulated theta: mu
        tabulated intensity: I
        """
        self.mu, self.I = self.get_theta(**kwargs)
        self.n_peaks = self.mu.shape[0]
        # Variables along the diffraction lines

        """
        parameters g, tau --> gamma, sigma^2
        """
        # Variables along the diffraction lines

        self.g = full((1, self.n_peaks), self.iw(1))
        self.tau = full((1, self.n_peaks), sigma)
        
    """
    Redefined variables
    """
    @staticmethod
    def iw(x):
        return (4 * x**2 -1) / (4 * x)

    @staticmethod
    def w(x):
        return 0.5 * (sqrt(x**2 + 1) + x)

    @staticmethod
    def der_w(x):
        return 0.5 * (x / sqrt(x**2 + 1) + 1)

    @property
    def gamma(self):
        return self.w(self.g)


    @staticmethod
    def u(x):
        return x**2

    @staticmethod
    def der_u(x):
        return 2 * x

    @property
    def sigma2(self):
        return self.u(self.tau)

    """
    Plot functions
    """
    def plot_spectra(self, *args, **kwargs):

        super().plot(*args, **kwargs)

    def plot(self, *args, **kwargs):

        plot(self.theta, self.z(), *args, **kwargs)

    """
    Utility functions
    """
    def get_theta(self,**kwargs):
        return self.phase.get_theta(**kwargs)

    def theta_range(self):
        return super().theta_range().squeeze()

    def z(self):
        """
        Synthetic spectrum.
        """

        self.precalculations()
        x = self.component_full.sum(axis = 1)
        self.del_precalculations()

        return x

    def z0(self):
        """
        Synthetic spectrum with gamma=1 for all peaks.
        """

        mu = self.mu[newaxis, :]
        I = self.I[newaxis, :]

        theta = self.theta[:,newaxis]
        component_core = exp((theta - mu)**2 / (-2 * self.sigma2))

        x = (I * component_core).sum(axis = 1)

        return x

    """
    Calculations for fit
    """
    def precalculations(self):

        # along the channels
        self.theta_calc = self.theta[:,newaxis]
        # along the diffraction lines
        #self.sigma2_calc = self.sigma2[:,newaxis]
        self.sigma2_calc = self.sigma2

        # along both axes
        mu = self.mu[newaxis,:]
        I = self.I[newaxis,:]

        self.component_core = exp((self.theta_calc - mu)**2 / (-2 * self.sigma2_calc))
        self.component_full = I * self.gamma * self.component_core

    def del_precalculations(self):

        del self.theta_calc
        del self.sigma2_calc
        del self.component_full
        del self.component_core

    def der_f_a_s_beta(self):

        mu = self.mu[newaxis,:]
        channel = self.channel[:, newaxis]

        der_theta_a = (180 / pi) * self.opt[1] / ((channel + self.opt[0])**2 + self.opt[1]**2)
        der_theta_s = (-180 / pi) * (channel + self.opt[0]) / ((channel + self.opt[0])**2 + self.opt[1]**2)

        aux = (self.component_full * (self.theta_calc - mu) / self.sigma2_calc).sum(axis = 1, keepdims = True)
        der_f_a = - der_theta_a * aux
        der_f_s = - der_theta_s * aux
        der_f_beta = - aux

        return der_f_a, der_f_s, der_f_beta

    def der_f_a_beta__when_relation_a_s(self, k, b):

        mu = self.mu[newaxis,:]
        channel = self.channel[:, newaxis]

        der_theta_a = (180 / pi) * (b - k * channel) / ( (channel + self.opt[0])**2 + (k * self.opt[0] + b)**2 )
        aux = (self.component_full * (self.theta_calc - mu) / self.sigma2_calc).sum(axis = 1, keepdims = True)
        der_f_a = - der_theta_a * aux
        der_f_beta = - aux

        return der_f_a, der_f_beta


    def der_f_g(self):

        I = self.I[newaxis,:]

        return I * self.component_core * self.der_w(self.g)

    def der_f_tau(self):

        mu = self.mu[newaxis,:]

        return self.component_full * ((self.theta_calc - mu)**2 / (2 * self.sigma2_calc**2)) * self.der_u(self.tau)

    def evolution_of_parameters(self):

        #self.intensity = spectrum.intensity#[:, newaxis]
        y = self.intensity[:,newaxis]
        f = self.component_full.sum(axis = 1, keepdims = True)
        r = y - f

        try:
            evol = pinv(self.Jacobian_f) @ r # or scipy.linalg.pinv
        except:
            evol = full((self.Jacobian_f.shape[1], 1), 0)
        finally:
            return evol


    def fit(self, k = None, b = None, a = False, s = False, beta = False, gamma = False, sigma = False, alpha = 1):
        """
        Performs a step of Gauss-Newton optimization. You need to choose the parameters that will be used to optimize. The other ones will be kept fixed.
        If you set k and b, parameters a and s are used in optimization (you don't need to explicitly set them to True) and are tied by the relation given by k and b.
        """

        is_used_relation = ((k is not None) and (b is not None))
        if is_used_relation:
            a = True
            s = False
        n_opt = a + s + beta
        n_gamma = gamma * self.n_peaks
        #n_sigma = sigma * self.n_peaks

        self.precalculations()
        Jacobian_construction = []

        # Calibration parameters
        if is_used_relation:
            der_f_a, der_f_beta = self.der_f_a_beta__when_relation_a_s(k, b)
        else:
            if (n_opt > 0):
                der_f_a, der_f_s, der_f_beta = self.der_f_a_s_beta()


        if a:
            Jacobian_construction.append(der_f_a)
        if s:
            Jacobian_construction.append(der_f_s)
        if beta:
            Jacobian_construction.append(der_f_beta)

        # Gamma
        if gamma:
            Jacobian_construction.append(self.der_f_g())

        # Sigma
        if sigma:
            Jacobian_construction.append(self.der_f_tau())

        # Jacobian
        if Jacobian_construction:
            self.Jacobian_f = concatenate(Jacobian_construction, axis = 1)

        else:
            self.del_precalculations()
            return

        # Evolution of parameters
        d_params = alpha * self.evolution_of_parameters()
        mask_opt = [a, s, beta]
        self.opt[mask_opt] += d_params[0:n_opt, 0]
        if is_used_relation:
            self.opt[1] = k * self.opt[0] + b
        if gamma:
            self.g += d_params[n_opt : (n_opt + n_gamma)].T
        if sigma:
            self.tau += d_params[(n_opt + n_gamma) :].T

        self.del_precalculations()

        del self.Jacobian_f

        return self

    def fit_cycle(self, max_steps = 8, **kwargs):
        for i in range(max_steps):
            self.fit(**kwargs)

        return self

    """
    Evaluation of the results
    """
    def plot_spectrum(self,*args,**kwargs):
        super().plot(*args,**kwargs)

    def plot(self,*args,**kwargs):
        plot(self.theta,self.z(),*args,**kwargs)

    def overlap(self):

        m = minimum(self.z(), self.intensity)
        m = where(m < 0, 0, m)

        return m

    def overlap_area(self):

        return self.overlap().sum()
