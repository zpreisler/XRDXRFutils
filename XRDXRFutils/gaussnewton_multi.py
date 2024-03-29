from .database import Phase, PhaseList
from .spectra import FastSpectraXRD
from .gaussnewton import GaussNewton
from .gammasearch import GammaMap

from numpy import (fabs, sum, exp, log, sin, pi, array, ones, zeros, full, full_like, trapz, fromiter,
    minimum, maximum, nanmax, std, sign, sqrt, square, average, clip, newaxis, concatenate, stack,
    append, where, arange, deg2rad, rad2deg, argsort, delete)
from numpy.linalg import pinv, inv

from matplotlib.pyplot import plot



class GaussNewton_MultiPhases(GaussNewton):
    """
    Class to calculate Gauss-Newton minimization of the synthetic and the experimental spectrum.
    It performs the minimization with multiple phases that share calibration parameters s, beta and have separate a.
    """

    def __init__(self, phases, spectrum, sigma = 0.2, clean_peaks = None, **kwargs):
        """
        Initialization of GaussNewton
        - phases: (list of Phase or of PhaseList)
            Tabulated phase.
        - spectrum: (SpectraXRD class)
            Experimental spectrum.
        - sigma: (float)
            Standard deviation of Gaussian peaks of the synthetic XRD patterns. Default is 0.2.
        - clean_peaks: (bool)
            Threshold used to clean pair of tabulated peaks that are close to each other.
            The default value is None, in which case no cleaning is performed.
        - kwargs: (different types, optional)
            Arguments to select peaks; they are passed to Phase.get_theta().
        """
        for phase in phases:
            if type(phase) not in [Phase, PhaseList]:
                raise Exception('GaussNewton initialization: invalid phase type.')

        self.phases = phases
        self.n_phases = len(phases)
        self.label = ' + '.join([p.label for p in phases])
        self.spectrum = spectrum
        self.opt = spectrum.opt[ self.n_phases * [0] + [1, 2] ].copy()
        self.kwargs = kwargs

        ### Variables along the diffraction lines ###
        # tabulated theta: mu
        # tabulated intensity: I
        # parameters g, tau --> gamma, sigma^2

        self.mu, self.I = [], []
        for idx in range(self.n_phases):
            mu, I, p = self.get_theta_partial(idx)
            self.mu.append(mu)
            self.I.append(I)
        if (clean_peaks is not None):
            self.clean_tab_peaks(clean_peaks)

        self.g = [full(m, self.iw(1)) for m in self.n_peaks]
        self.tau = [full(m, self.iu(sigma**2)) for m in self.n_peaks]


    def clean_tab_peaks(self, threshold):
        mu = concatenate([arr for arr in self.mu])
        I = concatenate([arr for arr in self.I])
        i_phase = concatenate([len(arr) * [i] for i, arr in enumerate(self.mu)])

        # Sort by tabulated angle
        idx_sorted = argsort(mu)
        mu, I, i_phase = mu[idx_sorted], I[idx_sorted], i_phase[idx_sorted]

        # Find pairs of very close peaks and delete one of the two
        j_delete = []
        for j in range(len(mu) - 1):
            if ((mu[j + 1] - mu[j]) < threshold):
                if (I[j + 1] > I[j]):
                    j_delete.append(j)
                else:
                    j_delete.append(j + 1)
        mu = delete(mu, j_delete)
        I = delete(I, j_delete)
        i_phase = delete(i_phase, j_delete)

        # Assign mu, I
        self.mu = [mu[(i_phase == idx)] for idx in range(self.n_phases)]
        self.I = [I[(i_phase == idx)] for idx in range(self.n_phases)]


    ### Redefined variables ###

    @property
    def gamma(self):
        return [self.w(g) for g in self.g]

    @property
    def sigma2(self):
        return [self.u(tau) for tau in self.tau]


    ### Utility functions ###

    @property
    def a(self):
        return self.opt[:self.n_phases]

    @property
    def s(self):
        return self.opt[self.n_phases]

    @property
    def beta(self):
        return self.opt[self.n_phases + 1]


    @property
    def n_peaks(self):
        """Number of tabulated peaks of each phase."""
        return [mu.shape[0] for mu in self.mu]


    def get_theta_partial(self, idx):
        """Tabulated peaks of the chosen phase."""
        return self.phases[idx].get_theta(**self.kwargs)


    @property
    def theta(self):
        """Angles corresponding to channels, according to calibration of each phase."""
        return [
            self.spectrum.fce_calibration(
                self.channel,
                * self.opt[[idx, self.n_phases, self.n_phases + 1]]
            ) for idx in range(self.n_phases)
        ]

    @property
    def theta_range(self):
        """Angular range, according to calibration of each phase."""
        return [
            self.spectrum.fce_calibration(
                array([self.channel[0], self.channel[-1]]),
                * self.opt[[idx, self.n_phases, self.n_phases + 1]]
            ) for idx in range(self.n_phases)
        ]


    def synthetic_spectrum_partial(self, idx, rescale_peaks):
        mu = self.mu[idx][newaxis, :]
        I = self.I[idx][newaxis, :]
        sigma2 = self.sigma2[idx][newaxis, :]
        theta = self.theta[idx][:, newaxis]

        component_core = exp((theta - mu)**2 / (-2 * sigma2))
        if (rescale_peaks):
            gamma = self.gamma[idx][newaxis, :]
            component_full = I * gamma * component_core
        else:
            component_full = I * component_core
        return component_full.sum(axis = 1)

    def z0_partial(self, idx):
        """Synthetic spectrum of the chosen phase, with gamma = 1 for all peaks."""
        return self.synthetic_spectrum_partial(idx, False)

    def z_partial(self, idx):
        """Synthetic spectrum of the chosen phase."""
        return self.synthetic_spectrum_partial(idx, True)


    def synthetic_spectrum_multiphase(self, rescale_peaks):
        mu = [arr[newaxis, :] for arr in self.mu]
        I = [arr[newaxis, :] for arr in self.I]
        sigma2 = [arr[newaxis, :] for arr in self.sigma2]
        theta = [arr[:, newaxis] for arr in self.theta]

        component_core = [exp((theta[idx] - mu[idx])**2 / (-2 * sigma2[idx])) for idx in range(self.n_phases)]
        if (rescale_peaks):
            gamma = [arr[newaxis, :] for arr in self.gamma]
            component_full = [(I[idx] * gamma[idx] * component_core[idx]).sum(axis = 1, keepdims = True) for idx in range(self.n_phases)]
        else:
            component_full = [(I[idx] * component_core[idx]).sum(axis = 1, keepdims = True) for idx in range(self.n_phases)]
        return concatenate(component_full, axis = 1)

    def z0_multiphase(self):
        """Synthetic spectrum of each phase, with gamma = 1 for all peaks."""
        return self.synthetic_spectrum_multiphase(False).T

    def z_multiphase(self):
        """Synthetic spectrum of each phase."""
        return self.synthetic_spectrum_multiphase(True).T


    def z0(self):
        """Synthetic spectrum of the combination of phases, with gamma = 1 for all peaks."""
        return self.synthetic_spectrum_multiphase(False).sum(axis = 1)

    def z(self):
        """Synthetic spectrum of the combination of phases."""
        return self.synthetic_spectrum_multiphase(True).sum(axis = 1)


    ### Calculations for fit ###

    def calculate_components(self):
        mu = [arr[newaxis, :] for arr in self.mu]
        I = [arr[newaxis, :] for arr in self.I]
        gamma = [arr[newaxis, :] for arr in self.gamma]
        sigma2 = [arr[newaxis, :] for arr in self.sigma2]
        theta = [arr[:, newaxis] for arr in self.theta]

        self.component_core = [exp((theta[idx] - mu[idx])**2 / (-2 * sigma2[idx])) for idx in range(self.n_phases)]
        self.component_full = [I[idx] * gamma[idx] * self.component_core[idx] for idx in range(self.n_phases)]

    def del_components(self):
        del self.component_core
        del self.component_full


    def der_f_a_s_beta(self, bool_a, bool_s, bool_beta):
        mu = [arr[newaxis, :] for arr in self.mu]
        sigma2 = [arr[newaxis, :] for arr in self.sigma2]
        channel = self.channel[:, newaxis]
        theta = [arr[:, newaxis] for arr in self.theta]

        a = self.opt[:self.n_phases]
        s = self.opt[self.n_phases]
        J = []
        aux = [(self.component_full[idx] * (theta[idx] - mu[idx]) / sigma2[idx]).sum(axis = 1, keepdims = True) for idx in range(self.n_phases)]

        if (sum(bool_a) + bool_s > 0):
            der_denominator = [(channel + a[idx])**2 + s**2 for idx in range(self.n_phases)]

            if (sum(bool_a) > 0):
                aux__sel = [e for idx, e in enumerate(aux) if bool_a[idx]]
                der_theta_a__sel = [rad2deg(s / e) for idx, e in enumerate(der_denominator) if bool_a[idx]]
                der_f_a = [- der_theta_a__sel[idx] * aux__sel[idx] for idx in range(len(aux__sel))]
                J += der_f_a

            if (bool_s):
                der_theta_s = [- rad2deg((channel + a[idx]) / der_denominator[idx]) for idx in range(self.n_phases)]
                der_f_s = - concatenate([der_theta_s[idx] * aux[idx] for idx in range(self.n_phases)], axis = 1).sum(axis = 1, keepdims = True)
                J += [der_f_s]

        if (bool_beta):
            der_f_beta = - concatenate(aux, axis = 1).sum(axis = 1, keepdims = True)
            J += [der_f_beta]

        return J


    def der_f_g(self):
        I = [arr[newaxis, :] for arr in self.I]
        return [I[idx] * self.component_core[idx] * self.der_w(self.g[idx]) for idx in range(self.n_phases)]


    def der_f_tau(self):
        mu = [arr[newaxis, :] for arr in self.mu]
        sigma2 = [arr[newaxis, :] for arr in self.sigma2]
        theta = [arr[:, newaxis] for arr in self.theta]
        return [self.component_full[idx] * ((theta[idx] - mu[idx])**2 / (2 * sigma2[idx]**2)) * self.der_u(self.tau[idx]) for idx in range(self.n_phases)]


    def fit(self, a = False, s = False, beta = False, gamma = False, sigma = False, alpha = 1, downsample = None):
        """
        Performs a step of Gauss-Newton optimization.
        You need to choose the parameters that will be used to optimize. The other ones will be kept fixed.
        - a: (bool or list of bools)
            Whether to fit for parameters a.
            If it is a single bool, it will be applied to all the phases of the given GaussNewton_MultiPhases instance.
            Otherwise, it can be a list of bools of the same length as the list of phases of the given GaussNewton_MultiPhases instance.
            In this case, each value of a is applied to the corresponding phase.
        """

        def f(self, a, s, beta, gamma, sigma, alpha):
            if (sum(self.n_peaks) > 0):

                if (sum(self.n_peaks) == 1):
                    s = False

                #check for a
                if (type(a) == bool):
                    a = self.n_phases * [a]
                elif ((type(a) != list) or (len(a) != self.n_phases)):
                    raise Exception('GaussNewton_MultiPhases.fit(): parameter a is invalid. It can be a bool or a list of bools of the same length as the list of phases of the given GaussNewton_MultiPhases instance.')

                n_opt = sum(a) + s + beta
                n_gamma = [n_peaks * gamma for n_peaks in self.n_peaks]
                n_sigma = [n_peaks * sigma for n_peaks in self.n_peaks]

                self.calculate_components()

                ### Construction of Jacobian ###
                Jacobian_construction = []

                # Calibration parameters
                if (n_opt > 0):
                    Jacobian_construction.extend(self.der_f_a_s_beta(a, s, beta))

                # Gamma
                if gamma:
                    Jacobian_construction.extend(self.der_f_g())

                # Sigma
                if sigma:
                    Jacobian_construction.extend(self.der_f_tau())

                # Jacobian
                Jacobian_f = concatenate(Jacobian_construction, axis = 1)

                ### Evolution of parameters ###
                y = self.intensity[:, newaxis]
                f = concatenate(self.component_full, axis = 1).sum(axis = 1, keepdims = True)
                r = y - f
                try:
                    evol = pinv(Jacobian_f) @ r
                except:
                    evol = full((Jacobian_f.shape[1], 1), 0)
                d_params = alpha * evol.squeeze()

                # Add evolution
                mask_opt = a + [s, beta]
                self.opt[mask_opt] += d_params[0 : n_opt]

                if gamma:
                    j = n_opt
                    for idx in range(self.n_phases):
                        self.g[idx] += d_params[j : (j + n_gamma[idx])]
                        j += n_gamma[idx]

                if sigma:
                    j = n_opt + sum(n_gamma)
                    for idx in range(self.n_phases):
                        self.tau[idx] += d_params[j : (j + n_sigma[idx])]
                        j += n_sigma[idx]

                self.del_components()

            return self

        return self.downsampled_function(downsample, f, a = a, s = s, beta = beta, gamma = gamma, sigma = sigma, alpha = alpha)


    def search(self, alpha = 1):
        self.fit_cycle(steps = 3, a = False, s = False, gamma = True, alpha = alpha, downsample = 3)
        self.fit_cycle(steps = 3, a = True, s = True, gamma = True, alpha = alpha, downsample = 2)
        self.fit_cycle(steps = 3, a = True, s = True, gamma = True, alpha = alpha, downsample = 1)
        self.fit_cycle(steps = 3, a = True, s = True, gamma = True, alpha = alpha, downsample = 0)
        return self


    ### Evaluation of the results ###

    def overlap_partial(self, idx, downsample = None):
        def f(self):
            return maximum(minimum(self.z_partial(idx), self.intensity), 0)

        return self.downsampled_function(downsample, f)


    ### Plot functions ###

    def plot_phase(self, idx, **kwargs):
        self.phases[idx].plot(**self.kwargs, **kwargs)



class GammaMap_MultiPhases(GammaMap):
    """
    Map that fits multiple phases in every pixel of the given data.
    The basic structure is a list of GaussNewton_MultiPhases objects, one for each pixel.
    """

    def __init__(self, list_elements = []):
        super().__init__(list_elements, GaussNewton_MultiPhases)


    def from_data(self, data, phases, indices_sel = None, sigma = 0.2, clean_peaks = None, **kwargs):
        """
        Builds the map that searches for given phases in given XRD data.
        
        Arguments
        ---------
        - data: (DataXRD)
            Contains the experimental XRD patterns for every pixel.
        - phases: (list of Phase)
            Phases that will be fitted to experimental XRD patterns.
        - indices_sel: (numpy array)
            2d numpy array of boolean type, of the same dimensions as data, telling for each pixel if it is included or not in the map.
            The default value is None, in which case all the pixels are included.
        - sigma: (float)
            Standard deviation of Gaussian peaks of the synthetic XRD patterns. Default is 0.2.
        - clean_peaks: (bool)
            Threshold used to clean pair of tabulated peaks that are close to each other.
            The default value is None, in which case no cleaning is performed.
        - kwargs: (different types, optional)
            Arguments that will be passed down to Phase.get_theta().
            They put restrictions on which peaks of tabulated phases are chosen to build synthetic XRD patterns.
        """
        self.from_data__core(data, phases, indices_sel)

        self.coordinates = []
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                if self.indices_sel[y, x]:
                    self.coordinates.append((x, y))
                    spectrum = FastSpectraXRD().from_Data(data, x, y)
                    self += [self.type_of_elements(phases, spectrum, sigma, clean_peaks, **kwargs)]

        return self


    def search(self, verbose = True, alpha = 1):
        list_result = self.parallelized(verbose, self.type_of_elements.search, alpha = alpha)
        map = type(self)(list_result)
        map.set_attributes_from(self)
        return map
    