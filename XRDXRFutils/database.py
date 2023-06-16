#!/usr/bin/env python

from matplotlib.pyplot import plot, figure, subplots, xlim, ylim, vlines, legend, fill_between, cm, text

from numpy import (loadtxt, arcsin, sin, pi, array, asarray, argmin, minimum, nanmax, concatenate, delete,
    linspace, arange, empty, ones, zeros, full, newaxis, exp, argsort, isin)
from numpy.random import randint
from glob import glob
import warnings



class Phase(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.select_peaks(None)


    def __len__(self):
        return len(self['_pd_peak_intensity'][0])


    @property
    def label(self):
        for field in ['label', '_chemical_name_mineral', '_chemical_name_common', '_chemical_formula_sum']:
            if field in self:
                return self[field]
        return None


    @staticmethod
    def theta_from_d(d, l = 1.541874):
        return (360 / pi) * arcsin(l / (2 * d))

    @staticmethod
    def d_from_theta(theta, l = 1.541874):
        return l / (2 * sin(pi * theta / 360))


    def get_theta(self, length = [1.541874], scale = [1.0], min_theta = None, max_theta = None, min_intensity = None, first_n_peaks = None, distance_merge = None):
        # Check if arguments have different values compared to last call to this function
        if not (hasattr(self, 'length_last') and length == self.length_last
            and hasattr(self, 'scale_last') and scale == self.scale_last
            and hasattr(self, 'min_theta_last') and min_theta == self.min_theta_last
            and hasattr(self, 'max_theta_last') and max_theta == self.max_theta_last
            and hasattr(self, 'min_intensity_last') and min_intensity == self.min_intensity_last
            and hasattr(self, 'first_n_peaks_last') and first_n_peaks == self.first_n_peaks_last
            and hasattr(self, 'distance_merge_last') and distance_merge == self.distance_merge_last
            and hasattr(self, 'peaks_selected_last') and self.peaks_selected == self.peaks_selected_last
            and hasattr(self, 'theta') and hasattr(self, 'intensity') and hasattr(self, 'position')
        ):
            self.length_last = length
            self.scale_last = scale
            self.min_theta_last = min_theta
            self.max_theta_last = max_theta
            self.min_intensity_last = min_intensity
            self.first_n_peaks_last = first_n_peaks
            self.distance_merge_last = distance_merge
            self.peaks_selected_last = self.peaks_selected

            # Obtain list of peaks
            d, i = self['_pd_peak_intensity']
            # Avoid peaks with null intensity
            i_max = nanmax(i)
            mask = (i > (i_max * 1e-4))
            d, i = d[mask], i[mask]
            # Join peaks given by different wavelengths
            theta, intensity = [], []
            for l, s in zip(length, scale):
                theta += [self.theta_from_d(d, l)]
                intensity += [i * s]
            theta, intensity = concatenate(theta), concatenate(intensity)

            if len(theta) > 0:
                # Sort peaks by increasing theta
                idx_sorted = argsort(theta)
                theta, intensity = theta[idx_sorted], intensity[idx_sorted]

                # Merge peaks
                if distance_merge is not None:
                    weight = intensity.copy()
                    while len(theta) > 1:
                        theta_diff = theta[1:] - theta[:-1]
                        idx_min = argmin(theta_diff)
                        if (theta_diff[idx_min] <= distance_merge):
                            theta_point = (weight[idx_min] * theta[idx_min] + weight[idx_min + 1] * theta[idx_min + 1]) / (weight[idx_min] + weight[idx_min + 1])
                            #intensity_point = intensity[idx_min] + intensity[idx_min + 1]
                            # The merged peak has the same height as the combination of the two Gaussian peaks (less than the simple sum of the two heights)
                            intensity_point = (intensity[idx_min] * exp((theta_point - theta[idx_min])**2 / (-2 * distance_merge**2)) +
                                intensity[idx_min + 1] * exp((theta_point - theta[idx_min + 1])**2 / (-2 * distance_merge**2)))
                            weight_point = weight[idx_min] + weight[idx_min + 1]
                            theta[idx_min] = theta_point
                            intensity[idx_min] = intensity_point
                            weight[idx_min] = weight_point
                            theta = delete(theta, [idx_min + 1])
                            intensity = delete(intensity, [idx_min + 1])
                            weight = delete(weight, [idx_min + 1])
                        else:
                            break

                # Assign position to peaks based on decreasing intensity
                position = empty(len(theta), dtype = int)
                idx_sorted = argsort(intensity)[::-1]
                position[idx_sorted] = range(len(theta))

                # Select by angle, intensity and position
                mask = ones(len(theta), bool)
                if min_theta is not None:
                    mask &= (theta >= min_theta)
                if max_theta is not None:
                    mask &= (theta <= max_theta) 
                if min_intensity is not None:
                    mask &= (intensity >= (i_max * min_intensity))
                if first_n_peaks is not None:
                    mask &= (position < first_n_peaks)
                if (self.peaks_selected is not None) and (self.peaks_selected != []):
                    mask &= isin(position, self.peaks_selected)
                theta, intensity, position = theta[mask], intensity[mask], position[mask]

                # Rescale intensity
                if len(theta) > 0:
                    intensity /= intensity.max()

            # Assign attributes
            self.theta, self.intensity, self.position = theta, intensity, position

        return self.theta.copy(), self.intensity.copy(), self.position.copy()


    def select_peaks(self, peaks_selected = None):
        self.peaks_selected = peaks_selected
        return self


    def set_key(self, key, value):
        if value is None:
            self.pop(key, None)
        else:
            self[key] = value
        return self

    def set_label(self, label = None):
        return self.set_key('label', label)

    def set_name(self, name = None):
        return self.set_key('name', name)

    def set_point(self, point = None):
        return self.set_key('point', point)


    def plot(self, x_axis_is_channel = False, calibration_function_theta_to_channel = None, calibration_parameters = None,
        convolution = False, positions = False, colors = 'red', linestyles = 'dashed', label = None, lineheight = None,
        length = None, min_theta = None, max_theta = None, min_intensity = None, first_n_peaks = None, distance_merge = None, **kwargs):

        theta, intensity, position = self.get_theta(length=length,min_theta = min_theta, max_theta = max_theta, min_intensity = min_intensity, first_n_peaks = first_n_peaks, distance_merge = distance_merge)

        if x_axis_is_channel:
            if (calibration_function_theta_to_channel is None or calibration_parameters is None):
                raise Exception('If x_axis_is_channel is True, calibration_function and calibration_parameters need to be provided.')
            x_points = calibration_function_theta_to_channel(theta, *calibration_parameters)
        else:
            x_points = theta

        if label is None:
            label = self.label

        if lineheight is None:
            lineheight = intensity
        else:
            lineheight = full(intensity.shape, lineheight)

        if (convolution and (distance_merge is not None)):
            gamma = full((1, len(theta)), 1)
            sigma2 = full((1, len(theta)), distance_merge**2)
            mu = theta[newaxis, :]
            I = intensity[newaxis, :]
            theta_to_plot = arange(min_theta, max_theta, 0.01)[:, newaxis]
            component_core = exp((theta_to_plot - mu)**2 / (-2 * sigma2))
            component_full = I * gamma * component_core
            z = component_full.sum(axis = 1)
            if x_axis_is_channel:
                x_to_plot = calibration_function_theta_to_channel(theta_to_plot, *calibration_parameters)
            else:
                x_to_plot = theta_to_plot
            plot(x_to_plot, z)
        vlines(x_points, 0, lineheight, colors = colors, linestyles = linestyles, label = label, **kwargs)
        if positions:
            for i in range(len(x_points)):
                text(x_points[i], lineheight[i], f'{position[i]}', ha = 'center', va = 'bottom', fontsize = 'x-small')


    def save_cif(self, filename):

        with open(filename, 'w') as file:

            for field in ['_chemical_formula_sum', '_chemical_name_mineral', '_chemical_name_common', 'name']:
                if field in self:
                    file.write(field + "  '" + self[field] + "'\n")

            if 'point' in self:
                file.write('point  ' + format(self['point'], 'd') + '\n')

            file.write('loop_\n')
            file.write('_pd_peak_d_spacing\n')
            file.write('_pd_peak_intensity\n')
            for d, i in self['_pd_peak_intensity'].T:
                d = format(d, '.6f')
                i = format(i, '.2f')
                file.write('     ' + str(d) + f'{str(i):>14}' + '\n')



class PhaseList(list):

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        if 'label' in kwargs:
            self.set_label(kwargs.pop('label'))


    @property
    def label(self):
        if hasattr(self, 'label_arg'):
            return self.label_arg
        else:
            return '[' + ', '.join([elem.label for elem in self]) + ']'


    def random(self):
        idx = randint(self.__len__())
        return self[idx]


    def get_theta(self, **kwargs):
        theta = []
        intensity = []
        position = []
        for phase in self:
            t, i, p = phase.get_theta(**kwargs)
            theta += [t]
            intensity += [i]
            position += [p]
        return concatenate(theta), concatenate(intensity), concatenate(position)


    def set_label(self, label = None):
        if label is None:
            if hasattr(self, 'label_arg'):
                delattr(self, 'label_arg')
        else:
            self.label_arg = label

    def set_name(self, name = None):
        for phase in self:
            phase.set_name(name)
        return self

    def set_point(self, point = None):
        for phase in self:
            phase.set_point(point)
        return self


    def plot(self, positions = False, cmap = 'tab10', min_theta = None, max_theta = None, min_intensity = None, first_n_peaks = None, **kwargs):
        cmap_sel = cm.get_cmap(cmap)
        for i, phase in enumerate(self):
            idx_color = i % cmap_sel.N
            phase.plot(positions = positions, min_theta = min_theta, max_theta = max_theta, min_intensity = min_intensity,
                first_n_peaks = first_n_peaks, colors = cmap_sel(idx_color), **kwargs)


    def save_cif(self, filename):
        for phase in self:
            phase.save_cif(filename[:-4] + '_' + phase.label + '.cif')



class DatabaseXRD(dict):

    def read_cifs(self, path):
        filenames = sorted(glob(path + '/*.cif'))
        if not filenames:
            warnings.warn('No files found')

        i = 0
        for filename in filenames:
            phase = Phase(name = filename)

            with open(filename, 'r') as f:
                for line in f:
                    x = line.split()
                    if x:
                        y = x[0]

                        if y in ['_chemical_formula_sum', '_chemical_name_mineral', '_chemical_name_common']:
                            value = ' '.join(x[1:]).replace("'", '')
                            if value != '':
                                phase[y] = value

                        if y == 'name':
                            phase[y] = ' '.join(x[1:]).replace("'", '')

                        if y == 'point':
                            phase[y] = int(x[1])

                        if y == '_pd_peak_intensity':
                            z = loadtxt(f, unpack = True, dtype = float)
                            phase[y] = z

            key = phase.label
            if key in self:
                self[key] += PhaseList([phase])
            else:
                self[key] = PhaseList([phase], label = i)
                i += 1

        self.fix_chemical_formula()
        return self


    def fix_chemical_formula(self):
        for p_set in self.values():
            is_found_formula = False
            for i, p in enumerate(p_set):
                if '_chemical_formula_sum' in p:
                    is_found_formula = True
                    break
            if is_found_formula:
                for p in p_set:
                    if '_chemical_formula_sum' not in p:
                        p['_chemical_formula_sum'] = p_set[i]['_chemical_formula_sum']


    def random(self):
        x = list(self.values())
        idx = randint(len(x))
        return x[idx]
