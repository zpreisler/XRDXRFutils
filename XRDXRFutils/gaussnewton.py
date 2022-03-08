from .spectra import SpectraXRD

from numpy import exp, log, pi, array, ones, zeros, full, trapz, minimum, fabs, sign, sqrt, square, average, clip, newaxis, concatenate
from numpy.linalg import pinv, inv

from matplotlib.pyplot import plot

class GaussNewton(SpectraXRD):
    """
    Class to calculate Gauss-Newton minimization of the synthetic and the experimental spectrum.
    """
    def __init__(self, phase, spectrum, min_theta = 0, max_theta = 53, min_intensity = 0.05):
        """
        phase: tabulated phase; Phase or PhaseList class
        spectrum: experimental spectrum; Spectra class
        """
        super().__init__()

        self.phase = phase
        self.spectrum = spectrum
        self.min_theta = min_theta
        self.max_theta = max_theta
        self.min_intensity = min_intensity

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
        self.mu, self.I = self.get_theta(min_theta = min_theta, max_theta = max_theta, min_intensity = min_intensity)
        self.n_peaks = self.mu.shape[0]
        # Variables along the diffraction lines
        self.mu = self.mu[newaxis, :]
        self.I = self.I[newaxis, :]

        """
        parameters g, tau --> gamma, sigma^2
        """
        # Variables along the diffraction lines
        self.g = full((1, self.n_peaks), 0.75) # needed to have gamma = 1
        self.tau = full((1, self.n_peaks), 0.2) # needed to have sigma2 = 0.04


    """
    Utility functions
    """
    def get_theta(self,**kwargs):
        return self.phase.get_theta(**kwargs)


    def z(self):
        """
        Synthetic spectrum.
        """
        self.prepare_dimensional_data()
        return self.component_full.sum(axis = 1)

    def z0(self):
        """
        Synthetic spectrum without the rescalings of peaks.
        """
        self.prepare_dimensional_data()
        return (self.I * self.component_core).sum(axis = 1)


    """
    Redefined variables
    """
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
    def prepare_dimensional_data(self):
        # along the channels
        self.theta_calc = self.theta[:]
        # along the diffraction lines
        self.sigma2_calc = self.sigma2[:]
        # along both axes
        self.component_core = exp((self.theta_calc - self.mu)**2 / (-2 * self.sigma2_calc))
        self.component_full = self.I * self.gamma * self.component_core


    def der_f_a_s_beta(self):
        der_theta_a = (180 / pi) * self.opt[1] / ((self.channel + self.opt[0])**2 + self.opt[1]**2)
        der_theta_s = (-180 / pi) * (self.channel + self.opt[0]) / ((self.channel + self.opt[0])**2 + self.opt[1]**2)
        aux = (self.component_full * (self.theta_calc - self.mu) / self.sigma2_calc).sum(axis = 1, keepdims = True)
        der_f_a = - der_theta_a * aux
        der_f_s = - der_theta_s * aux
        der_f_beta = - aux
        return der_f_a, der_f_s, der_f_beta


    def der_f_g(self):
        return self.I * self.component_core * self.der_w(self.g)


    def der_f_tau(self):
        return self.component_full * ((self.theta_calc - self.mu)**2 / (2 * self.sigma2_calc**2)) * self.der_u(self.tau)


    def evolution_of_parameters(self):
        y = self.intensity
        f = self.component_full.sum(axis = 1, keepdims = True)
        r = y - f
        return (pinv(self.Jacobian_f) @ r) # or scipy.linalg.pinv


    def fit(self, a = False, s = False, beta = False, gamma = False, sigma = False, alpha = 1):
        """
        Performs a step of Gauss-Newton optimization. You need to choose the parameters that will be used to optimize. The other ones will be kept fixed.
        """
        self.prepare_dimensional_data()
        Jacobian_construction = []

        # Calibration parameters
        n_calibration = a + s + beta
        if (n_calibration > 0):
            der_f_a, der_f_s, der_f_beta = self.der_f_a_s_beta()
            if a:
                Jacobian_construction.append(der_f_a)
            if s:
                Jacobian_construction.append(der_f_s)
            if beta:
                Jacobian_construction.append(der_f_beta)

        # Gamma
        if gamma:
            n_gamma = self.n_peaks
            Jacobian_construction.append(self.der_f_g())
        else:
            n_gamma = 0

        # Sigma
        if sigma:
            Jacobian_construction.append(self.der_f_tau())

        # Jacobian
        if Jacobian_construction:
            self.Jacobian_f = concatenate(Jacobian_construction, axis = 1)
        else:
            return

        # Evolution of parameters
        d_params = alpha * self.evolution_of_parameters()
        if a:
            self.opt[0] += d_params[0, 0]
        if s:
            self.opt[1] += d_params[1, 0]
        if beta:
            self.opt[2] += d_params[2, 0]
        if gamma:
            self.g += d_params[n_calibration : (n_calibration + n_gamma)].T
        if sigma:
            self.tau += d_params[(n_calibration + n_gamma) :].T


    """
    Evaluation of the results
    """
    def plot_spectrum(self,*args,**kwargs):
        super().plot(*args,**kwargs)

    def plot(self,*args,**kwargs):
        plot(self.theta,self.z(),*args,**kwargs)


    def loss(self):
        return sum(square(self.intensity.squeeze() - self.z()))

    def fit_error(self):
        return sqrt(average(square(self.intensity.squeeze() - self.z())))


    def area_fit(self):
        return trapz(self.z())

    def area_0(self):
        return trapz(self.z0())


    def overlap(self):
        m =  minimum(self.z(), self.intensity.squeeze())
        m[m < 0] = 0
        return m

    def overlap_0(self):
        return trapz(minimum(self.z(), self.z0()))

    def overlap_area(self):
        return trapz(self.overlap())


    def overlap_ratio(self):
        integral_intersection = clip(self.z(), 0, self.intensity.squeeze()).sum()
        integral_data = self.intensity.sum()
        return (integral_intersection / integral_data)


    def component_ratio(self):
        gamma_adjusted = self.gamma**(-sign(self.gamma - 1))
        mask = ((self.mu >= self.theta.min()) & (self.mu <= self.theta.max()))
        
        #penalty = (self.I[mask] * gamma_adjusted[mask]).sum() / self.I[mask].sum()
        penalty = exp( (self.I[mask] * log(gamma_adjusted[mask])).sum() / self.I[mask].sum() )
        
        intersection_integral = clip(self.z(), 0, self.intensity.squeeze()).sum()
        data_integral = self.intensity.sum()
        return penalty * (intersection_integral / data_integral)
