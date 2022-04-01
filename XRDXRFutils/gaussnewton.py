from .spectra import SpectraXRD,FastSpectraXRD

from numpy import sum, exp, log, pi, array, ones, zeros, full, full_like, trapz, minimum, maximum, std, fabs, sign, sqrt, square, average, clip, newaxis, concatenate, append, where
from numpy.linalg import pinv, inv

from scipy.optimize import newton

from matplotlib.pyplot import plot

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

        self.channel = spectrum.channel[:, newaxis]
        self.intensity = spectrum.intensity[:, newaxis]

        """
        Phases

        tabulated theta: mu
        tabulated intensity: I
        """
        self.mu, self.I = self.get_theta(**kwargs)
        self.n_peaks = self.mu.shape[0]
        # Variables along the diffraction lines
        self.mu = self.mu[newaxis, :]
        self.I = self.I[newaxis, :]

        """
        parameters g, tau --> gamma, sigma^2
        """
        self.g = full((1, self.n_peaks), self.iw(1))
        self.tau = full((1, self.n_peaks), sigma)

    """
    Plot functions
    """
    def plot_spectrum(self, *args, **kwargs):
        #super().plot(*args, **kwargs)
        plot(self.theta.squeeze(), self.intensity.squeeze(), *args, **kwargs)

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
        component_core = exp((self.theta - self.mu)**2 / (-2 * self.sigma2))
        x = (self.I * component_core).sum(axis = 1)

        return x


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
    Calculations for fit
    """
    def precalculations(self):

        # along the channels
        self.theta_calc = self.theta
        # along the diffraction lines
        self.sigma2_calc = self.sigma2
        # along both axes

        self.component_core = exp((self.theta_calc - self.mu)**2 / (-2 * self.sigma2_calc))
        self.component_full = self.I * self.gamma * self.component_core

    def del_precalculations(self):

        del self.theta_calc
        del self.sigma2_calc
        del self.component_full
        del self.component_core

    def der_f_a_s_beta(self):

        der_theta_a = (180 / pi) * self.opt[1] / ((self.channel + self.opt[0])**2 + self.opt[1]**2)
        der_theta_s = (-180 / pi) * (self.channel + self.opt[0]) / ((self.channel + self.opt[0])**2 + self.opt[1]**2)

        aux = (self.component_full * (self.theta_calc - self.mu) / self.sigma2_calc).sum(axis = 1, keepdims = True)
        der_f_a = - der_theta_a * aux
        der_f_s = - der_theta_s * aux
        der_f_beta = - aux

        return der_f_a, der_f_s, der_f_beta

    def der_f_a_beta__when_relation_a_s(self, k, b):

        der_theta_a = (180 / pi) * (b - k * self.channel) / ( (self.channel + self.opt[0])**2 + (k * self.opt[0] + b)**2 )
        aux = (self.component_full * (self.theta_calc - self.mu) / self.sigma2_calc).sum(axis = 1, keepdims = True)
        der_f_a = - der_theta_a * aux
        der_f_beta = - aux

        return der_f_a, der_f_beta

    def der_f_g(self):
        return self.I * self.component_core * self.der_w(self.g)

    def der_f_tau(self):
        return self.component_full * ((self.theta_calc - self.mu)**2 / (2 * self.sigma2_calc**2)) * self.der_u(self.tau)

    def evolution_of_parameters(self):
        y = self.intensity
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
            del Jacobian_construction

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

    def fit_cycle(self, max_steps = 16, error_tolerance = 1e-4, **kwargs):
        fit_errors = array([])
        for i in range(max_steps):
            self.fit(**kwargs)
            if (error_tolerance is not None):
                fit_errors = append(fit_errors, self.fit_error())
                if (i >= 3):
                    if (std(fit_errors[-4:]) < error_tolerance):
                        break

    """
    Evaluation of the results
    """
    def plot(self,*args,**kwargs):
        plot(self.theta,self.z(),*args,**kwargs)

    def loss(self):
        return ((self.intensity.squeeze() - self.z())**2).mean()

    def loss_0(self):
        return ((self.spectrum.rescaling * (self.spectrum.intensity - self.z()))**2).mean()

    def fit_error(self):
        return sqrt(average(square(self.intensity.squeeze() - self.z())))
        #return sqrt(((self.intensity.squeeze() - self.z())**2).mean())

    def area_fit(self):
        return self.z().sum()

    def area_0(self):
        return self.z0().sum()

    def area_min_0_fit(self):
        return minimum(self.z(), self.z0()).sum()

    def overlap(self):
        m =  minimum(self.z(), self.intensity.squeeze())
        m[m < 0] = 0
        return m

    def overlap_area(self):
        return self.overlap().sum()

    def overlap_ratio(self):
        integral_intersection = self.overlap().sum()
        integral_spectrum = clip(self.intensity, 0, None).sum()
        return (integral_intersection / integral_spectrum)

    def fit_penalty(self):
        z0 = self.z0()
        z = self.z()

        rescaling = where(z0 > 1e-3, z / z0, 1)
        rescaling_adjusted = rescaling**(-sign(rescaling - 1))

        return exp( (z0 * log(rescaling_adjusted)).sum() / z0.sum() )

    def component_ratio(self):
        return self.overlap_ratio() * self.fit_penalty()
