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

        self.label = phase.label

        self.min_theta = min_theta
        self.max_theta = max_theta
        self.min_intensity = min_intensity

        self.phase = phase
        self.spectrum = spectrum

        """
        Spectrum
        """
        self.opt = spectrum.opt.copy()
        self.channel = spectrum.channel
        self.intensity = spectrum.intensity

        """
        Phases

        tabulated theta: mu
        tabulated intensity: I
        """
        self.mu, self.I = self.get_theta(min_theta = min_theta, max_theta = max_theta, min_intensity = min_intensity)

        """
        parameters tau, g --> sigma^2, gamma
        """
        #self.tau = full(len(self.mu), -6.21) # needed to have sigma2 = 0.04
        self.tau = full(len(self.mu), 0.04) # needed to have sigma2 = 0.04
        self.g = ones(len(self.I)) * 0.75 # needed to have gamma = 1


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
        return (self.I_dim * self.component_core).sum(axis = 1)


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


    # @staticmethod
    # def u(x):
    #     return GaussNewton.w(100 * x) / 100

    # @staticmethod
    # def der_u(x):
    #     return GaussNewton.der_w(100 * x)

    @staticmethod
    def u(x):
        return x

    @staticmethod
    def der_u(x):
        return 1

    @property
    def sigma2(self):
        return self.u(self.tau)


    """
    Calculations for fit
    """
    def prepare_dimensional_data(self):
        # Variables along the channels
        self.channel_dim = self.channel[:, newaxis]
        self.theta_dim = self.theta[:, newaxis]

        # Variables along the diffraction lines
        self.mu_dim = self.mu[newaxis, :]
        self.I_dim = self.I[newaxis, :]
        self.g_dim = self.g[newaxis, :]
        self.gamma_dim = self.gamma[newaxis, :]
        self.tau_dim = self.tau[newaxis, :]
        self.sigma2_dim = self.sigma2[newaxis, :]

        # Components of the synthetic function (they go along both axes)
        self.component_core = exp((self.theta_dim - self.mu_dim)**2 / (-2 * self.sigma2_dim))
        self.component_full = self.I_dim * self.gamma_dim * self.component_core


    def der_f_a_s_beta(self):
        der_theta_a = (180 / pi) * self.opt[1] / ((self.channel_dim + self.opt[0])**2 + self.opt[1]**2)
        der_theta_s = (-180 / pi) * (self.channel_dim + self.opt[0]) / ((self.channel_dim + self.opt[0])**2 + self.opt[1]**2)
        aux = (self.component_full * (self.theta_dim - self.mu_dim) / self.sigma2_dim).sum(axis = 1, keepdims = True)
        der_f_a = - der_theta_a * aux
        der_f_s = - der_theta_s * aux
        der_f_beta = - aux
        return der_f_a, der_f_s, der_f_beta


    def der_f_g(self):
        return self.I_dim * self.component_core * self.der_w(self.g_dim)


    def der_f_tau(self):
        #f_dim = self.component_full.sum(axis = 1, keepdims = True)
        #return f_dim * ((self.theta_dim - self.mu_dim)**2 / (2 * (self.sigma2_dim)**2)) * self.der_u(self.tau_dim)
        return self.I_dim * self.component_core * self.w(self.g_dim) * 0.5 * ((self.theta_dim - self.mu_dim)**2 / (2 * (self.sigma2_dim)**2)) 


    def evolution_of_parameters(self):
        y = self.intensity
        f = self.component_full.sum(axis = 1)
        r = y - f
        return (pinv(self.Jacobian_f) @ r) # or scipy.linalg.pinv


    def fit_GN_a_s_beta(self, alpha = 1):
        """
        Optimizes on a, s, beta. Keeps gamma and sigma fixed.
        """
        self.prepare_dimensional_data()

        # Derivatives with respect to a, s, beta
        der_f_a, der_f_s, der_f_beta = self.der_f_a_s_beta()

        # Jacobian
        self.Jacobian_f = concatenate((der_f_a, der_f_s, der_f_beta), axis = 1)

        # Evolution of parameters
        d_params = alpha * self.evolution_of_parameters()
        self.opt += d_params


    def fit_GN_a_s_gamma(self, alpha = 1):
        """
        Optimizes on a, s, gamma. Keeps beta and sigma fixed.
        """
        self.prepare_dimensional_data()

        # Derivatives with respect to a, s
        der_f_a, der_f_s, der_f_beta = self.der_f_a_s_beta()

        # Jacobian
        self.Jacobian_f = concatenate((der_f_a, der_f_s, self.der_f_g()), axis = 1)

        # Evolution of parameters
        d_params = alpha * self.evolution_of_parameters()
        self.opt[:2] += d_params[:2]
        self.g += d_params[2:]


    def fit_GN_gamma(self, alpha = 1):
        """
        Optimizes on gamma. Keeps all the other parameters fixed.
        """
        self.prepare_dimensional_data()

        # Jacobian
        self.Jacobian_f = self.der_f_g()

        # Evolution of parameters
        d_params = alpha * self.evolution_of_parameters()
        self.g += d_params


    def fit_GN_sigma(self, alpha = 1):
        """
        Optimizes on sigma. Keeps all the other parameters fixed.
        """
        self.prepare_dimensional_data()

        # Jacobian
        self.Jacobian_f = self.der_f_tau()

        # Evolution of parameters
        d_params = alpha * self.evolution_of_parameters()
        self.tau += d_params


    """
    Evaluation of the results
    """
    def plot_spectrum(self,*args,**kwargs):
        super().plot(*args,**kwargs)

    def plot(self,*args,**kwargs):
        plot(self.theta,self.z(),*args,**kwargs)


    def loss(self):
        return sum(square(self.intensity - self.z()))

    def fit_error(self):
        return sqrt(average(square(self.intensity - self.z())))


    def area_fit(self):
        return trapz(self.z())

    def area_0(self):
        return trapz(self.z0())


    def overlap(self):
        m =  minimum(self.z(), self.intensity)
        m[m < 0] = 0
        return m

    def overlap_0(self):
        return trapz(minimum(self.z(), self.z0()))

    def overlap_area(self):
        return trapz(self.overlap())


    def overlap_ratio(self):
        integral_intersection = clip(self.z(), 0, self.intensity).sum()
        integral_data = self.intensity.sum()
        return (integral_intersection / integral_data)


    def component_ratio(self):
        gamma_adjusted = self.gamma**(-sign(self.gamma - 1))
        mask = ((self.mu >= self.theta.min()) & (self.mu <= self.theta.max()))
        
        #penalty = (self.I[mask] * gamma_adjusted[mask]).sum() / self.I[mask].sum()
        penalty = exp( (self.I[mask] * log(gamma_adjusted[mask])).sum() / self.I[mask].sum() )
        
        intersection_integral = clip(self.z(), 0, self.intensity).sum()
        data_integral = self.intensity.sum()
        return penalty * (intersection_integral / data_integral)
