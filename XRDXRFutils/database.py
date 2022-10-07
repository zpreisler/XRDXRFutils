#!/usr/bin/env python

from matplotlib.pyplot import plot, figure, subplots, xlim, ylim, vlines, legend, fill_between, cm, text

from numpy import (loadtxt, arcsin, sin, pi, array, asarray, minimum, concatenate, linspace, arange,
    ones, zeros, full)
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


    def get_theta(self, length = [1.541874], scale = [1.0], min_theta = None, max_theta = None, min_intensity = None, first_n_peaks = None):

        if not (hasattr(self, 'length_last') and length == self.length_last
            and hasattr(self, 'scale_last') and scale == self.scale_last
            and hasattr(self, 'min_theta_last') and min_theta == self.min_theta_last
            and hasattr(self, 'max_theta_last') and max_theta == self.max_theta_last
            and hasattr(self, 'min_intensity_last') and min_intensity == self.min_intensity_last
            and hasattr(self, 'first_n_peaks_last') and first_n_peaks == self.first_n_peaks_last
            and hasattr(self, 'peaks_selected_last') and self.peaks_selected == self.peaks_selected_last
            and hasattr(self, 'theta') and hasattr(self, 'intensity')
        ):
            self.length_last = length
            self.scale_last = scale
            self.min_theta_last = min_theta
            self.max_theta_last = max_theta
            self.min_intensity_last = min_intensity
            self.first_n_peaks_last = first_n_peaks
            self.peaks_selected_last = self.peaks_selected

            d, i = self['_pd_peak_intensity']
            theta = []
            intensity = []
            for l, s in zip(length, scale):
                theta += [self.theta_from_d(d, l)]
                intensity += [i * s]
            theta = concatenate(theta)
            intensity = concatenate(intensity) / 1000.0
            intensity, theta = array(sorted(zip(intensity, theta), reverse = True)).T
            position = array(range(len(theta)))

            mask = ones(len(theta), bool)
            if min_theta is not None:
                mask &= (theta > min_theta)
            if max_theta is not None:
                mask &= (theta < max_theta) 
            if min_intensity is not None:
                mask &= (intensity > min_intensity)
            if first_n_peaks is not None:
                mask &= (position < first_n_peaks)
            if (self.peaks_selected is not None) and (self.peaks_selected != []):
                mask_peaks_selected = zeros(len(theta), bool)
                mask_peaks_selected[self.peaks_selected] = True
                mask &= mask_peaks_selected
            self.theta, self.intensity, self.position = theta[mask], intensity[mask], position[mask]
            self.theta, self.intensity, self.position = array(sorted(zip(self.theta, self.intensity, self.position))).T

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


    def plot(self, positions = False, colors = 'red', linestyles = 'dashed', label = None, lineheight = None,
         min_theta = None, max_theta = None, min_intensity = None, first_n_peaks = None, **kwargs):

        theta, intensity, position = self.get_theta(min_theta = min_theta, max_theta = max_theta, min_intensity = min_intensity, first_n_peaks = first_n_peaks)

        if label is None:
            label = self.label

        if lineheight is None:
            lineheight = intensity
        else:
            lineheight = full(intensity.shape, lineheight)

        vlines(theta, 0, intensity, colors = colors, linestyles = linestyles, label = label, **kwargs)
        if positions:
            for i in range(len(theta)):
                text(theta[i], lineheight[i], f'{position[i]:.0f}', ha = 'center', va = 'bottom', fontsize = 'x-small')


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
        super().__init__(*args, **kwargs)
        if 'label' in kwargs:
            self.label_arg = kwargs.pop('label')


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


    def set_name(self, name):
        for phase in self:
            phase.set_name(name)
        return self

    def set_point(self, point):
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
