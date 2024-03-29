from .spectra import SpectraXRD, FastSpectraXRD
from .database import Phase, PhaseList

from numpy import (fabs, sum, exp, log, sin, pi, array, ones, zeros, full, full_like, trapz, fromiter,
    minimum, maximum, nanmax, std, sign, sqrt, square, average, clip, newaxis, concatenate, stack,
    append, where, arange, deg2rad, rad2deg)
from numpy.linalg import pinv, inv

from matplotlib.pyplot import plot



class GaussNewton(FastSpectraXRD):
    """
    Class to calculate Gauss-Newton minimization of the synthetic and the experimental spectrum.
    """

    def __init__(self, phase, spectrum, sigma = 0.2, **kwargs):
        """
        Initialization of GaussNewton
        - phase: (Phase or PhaseList class)
            Tabulated phase.
        - spectrum: (SpectraXRD class)
            Experimental spectrum.
        - sigma: (float)
            Standard deviation of Gaussian peaks of the synthetic XRD patterns. Default is 0.2.
        - kwargs: (different types, optional)
            Arguments to select peaks; they are passed to Phase.get_theta().
        """
        if type(phase) not in [Phase, PhaseList]:
            raise Exception('GaussNewton initialization: invalid phase type.')

        self.phase = phase
        self.spectrum = spectrum
        self.kwargs = kwargs
        self.label = phase.label
        self.opt = spectrum.opt.copy()

        ### Variables along the diffraction lines ###
        # tabulated theta: mu
        # tabulated intensity: I
        # parameters g, tau --> gamma, sigma^2
        self.mu, self.I, p = self.get_theta()
        self.g = full((1, self.n_peaks), self.iw(1))
        self.tau = full((1, self.n_peaks), self.iu(sigma**2))


    ### Redefined variables ###
    @staticmethod
    def w(x):
        return 0.5 * (sqrt(x**2 + 1) + x)

    @staticmethod
    def iw(x):
        return (4 * x**2 -1) / (4 * x)

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
    def iu(x):
        return sqrt(x)

    @staticmethod
    def der_u(x):
        return 2 * x

    @property
    def sigma2(self):
        return self.u(self.tau)


    ### Downsample functions ###

    def downsample(self, level):
        self.spectrum.downsample(level)
        return self

    @property
    def downsample_level(self):
        return self.spectrum.downsample_level

    def downsampled_function(self, downsample, f, **kwargs):
        """
        Calls given method with chosen downsample.

        Arguments
        ---------
        - downsample: (int)
            Level of downsample
        - f: (method of GaussNewton)
            Method that will be called with the chosen downsample level.
        - kwargs: (different types, optional)
            Arguments that will be passed to f().

        Return
        ------
        The same as f(self, **kwargs), but calculated with the chosen downsample.
        """
        if downsample is not None:
            downsample_initial = self.downsample_level
            self.downsample(downsample)

        result = f(self, **kwargs)

        if downsample is not None:
            self.downsample(downsample_initial)

        return result


    ### Utility functions ###

    @property
    def channel(self):
        return self.spectrum.channel

    @property
    def intensity(self):
        return self.spectrum.intensity


    @property
    def n_peaks(self):
        """Number of tabulated peaks."""
        return self.mu.shape[0]

    def get_theta(self):
        """Tabulated peaks."""
        return self.phase.get_theta(**self.kwargs)


    @property
    def theta_range(self):
        """Angular range, according to calibration."""
        return super().theta_range()


    def z0(self):
        """Synthetic spectrum with gamma = 1 for all peaks."""
        mu = self.mu[newaxis, :]
        I = self.I[newaxis, :]
        theta = self.theta[:, newaxis]

        component_core = exp((theta - mu)**2 / (-2 * self.sigma2))
        x = (I * component_core).sum(axis = 1)
        return x

    def z(self):
        """Synthetic spectrum."""
        mu = self.mu[newaxis, :]
        I = self.I[newaxis, :]
        theta = self.theta[:, newaxis]

        component_core = exp((theta - mu)**2 / (-2 * self.sigma2))
        component_full = I * self.gamma * component_core
        x = component_full.sum(axis = 1)
        return x


    ### Calculations for fit ###

    def calculate_components(self):

        self.theta_calc = self.theta[:,newaxis]
        self.sigma2_calc = self.sigma2

        mu = self.mu[newaxis,:]
        I = self.I[newaxis,:]

        self.component_core = exp((self.theta_calc - mu)**2 / (-2 * self.sigma2_calc))
        self.component_full = I * self.gamma * self.component_core


    def del_components(self):
        del self.theta_calc
        del self.sigma2_calc
        del self.component_full
        del self.component_core


    def der_f_a_s_beta(self):

        mu = self.mu[newaxis,:]
        channel = self.channel[:, newaxis]

        a, s, beta = self.opt

        der_theta_a = rad2deg(s / ((channel + a)**2 + s**2))
        der_theta_s = - rad2deg((channel + a) / ((channel + a)**2 + s**2))

        aux = (self.component_full * (self.theta_calc - mu) / self.sigma2_calc).sum(axis = 1, keepdims = True)

        der_f_a = - der_theta_a * aux
        der_f_s = - der_theta_s * aux
        der_f_beta = - aux

        return der_f_a, der_f_s, der_f_beta


    def der_f_a_beta_when_relation_a_s(self, k, b):

        mu = self.mu[newaxis,:]
        channel = self.channel[:, newaxis]

        a, s, beta = self.opt

        der_theta_a = rad2deg((b - k * channel) / ( (channel + a)**2 + (k * a + b)**2 ))

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


    def fit(self, k = None, b = None, a = False, s = False, beta = False, gamma = False, sigma = False, alpha = 1, downsample = None):
        """
        Performs a step of Gauss-Newton optimization.
        You need to choose the parameters that will be used to optimize. The other ones will be kept fixed.
        If you set k and b, parameters a and s are used in optimization (you don't need to explicitly set them to True) and are tied by the relation given by k and b.
        """
        # To DO: Remove peaks that fall outside theta_range, because they can cause anomalous values

        def f(self, k, b, a, s, beta, gamma, sigma, alpha):
            if self.n_peaks > 0:

                if self.n_peaks == 1:
                    s = False

                is_used_relation = ((k is not None) and (b is not None))
                if is_used_relation:
                    a = True
                    s = False

                n_opt = a + s + beta
                n_gamma = gamma * self.n_peaks

                self.calculate_components()

                Jacobian_construction = []

                # Calibration parameters
                if is_used_relation:
                    der_f_a, der_f_beta = self.der_f_a_beta_when_relation_a_s(k, b)

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
                Jacobian_f = concatenate(Jacobian_construction, axis = 1)

                # Evolution of parameters
                y = self.intensity[:, newaxis]
                f = self.component_full.sum(axis = 1, keepdims = True)
                r = y - f
                try:
                    evol = pinv(Jacobian_f) @ r
                except:
                    evol = full((Jacobian_f.shape[1], 1), 0)
                d_params = alpha * evol

                mask_opt = [a, s, beta]
                self.opt[mask_opt] += d_params[0:n_opt, 0]

                if is_used_relation:
                    self.opt[1] = k * self.opt[0] + b

                if gamma:
                    self.g += d_params[n_opt : (n_opt + n_gamma)].T

                if sigma:
                    self.tau += d_params[(n_opt + n_gamma) :].T

                self.del_components()

            return self

        return self.downsampled_function(downsample, f, k = k, b = b, a = a, s = s, beta = beta, gamma = gamma, sigma = sigma, alpha = alpha)


    def fit_cycle(self, steps = 8, **kwargs):
        for i in range(steps):
            self.fit(**kwargs)
        return self


    ### Evaluation of the results ###

    def area(self):
        return self.z().sum()

    def area0(self):
        return self.z0().sum()


    def overlap(self, downsample = None):
        def f(self):
            return maximum(minimum(self.z(), self.intensity), 0)

        return self.downsampled_function(downsample, f)


    def overlap_area(self, downsample = None):
        return self.overlap(downsample).sum()


    def overlap_area_ratio(self, downsample = None):
        def f(self):
            intensity_corrected = maximum(self.intensity, 0)
            return self.overlap_area() / intensity_corrected.sum()

        return self.downsampled_function(downsample, f)


    def adjustment_ratio(self, downsample = None):
        def f(self):
            z0 = clip(self.z0(), None, 1) # to avoid anomalously high peaks resulting from overlapping tabulated peaks
            z = clip(self.z(), None, 1)
            intensity_corrected = maximum(self.intensity, 0)
            z_stack = stack((z0, z, intensity_corrected))
            z_min = z_stack.min(axis = 0)
            z_max = z_stack.max(axis = 0)
            return z_min.sum() / z_max.sum()

        return self.downsampled_function(downsample, f)


    def phase_presence(self, downsample = None, method = None, correction = None):
        # Default values
        if method is None:
            method = 'adjustment_ratio'
        if correction is None:
            correction = False

        methods_allowed = ['overlap_area', 'overlap_area_ratio', 'adjustment_ratio']
        if method in methods_allowed:
            result = getattr(self, method)(downsample)
            if correction:
                result *= (self.spectrum.rescaling**0.5)
        else:
            raise Exception('GaussNewton.phase_presence(): \'method\' argument has only some allowed values: ' + ', '.join(["'" + m + "'" for m in methods_allowed]) + '.')

        return result


    def L1loss(self, downsample = None):
        def f(self):
            return (fabs(self.intensity - self.z())).mean()

        return self.downsampled_function(downsample, f)


    def MSEloss(self, downsample = None):
        def f(self):
            return ((self.intensity - self.z())**2).mean()

        return self.downsampled_function(downsample, f)


    ### Plot functions ###

    def plot_spectrum(self, *args, **kwargs):
        super().plot(*args, **kwargs)

    def plot_phase(self, **kwargs):
        self.phase.plot(**self.kwargs, **kwargs)

    def plot(self, *args, **kwargs):
        plot(self.theta, self.z(), *args, **kwargs)


    ### Misc functions ###

    def make_phase(self):
        """
        Creates experimental phase from the phase used to create the given instance of GaussNewton.
        An instance of GaussNewton can be created with a Phase or a PhaseList passed as the argument 'phase'.
        According to that, this function returns a Phase or a PhaseList.

        Please, note that this function excludes peaks that lay outside the angle range of experimental signal.
        This is because they can have anomalously high value in order to fit the last bit of signal. This dwarfs all the other peaks.
        """
        if type(self.phase) == Phase:
            pl = PhaseList([self.phase])
        elif type(self.phase) == PhaseList:
            pl = self.phase

        pl_new = PhaseList([])
        index_count = 0

        for phase in pl:
            mu, I, p = self.get_theta()
            phase_len = mu.shape[0]

            gamma = self.gamma.squeeze()[index_count : (index_count + phase_len)]
            index_count += phase_len
            I_new = I * gamma

            # Selects only the peaks that lay inside the angle range of experimental signal.
            # Otherwise, external peaks can have anomalously high value in order to fit the last bit of signal. This dwarfs all the other peaks.
            theta_min, theta_max = self.theta_range
            mask_theta = ((mu >= theta_min) & (mu <= theta_max))
            mu = mu[mask_theta]
            I_new = I_new[mask_theta]

            # Calculates d, i
            I_new /= nanmax(I_new)
            i = I_new * 1000
            d = Phase.d_from_theta(mu)

            phase_new = Phase(phase)
            phase_new['_pd_peak_intensity'] = array([d, i])

            # Adds the new phase to the results
            pl_new.append(phase_new)

        if len(pl_new) == 1:
            return pl_new[0]
        else:
            return pl_new
