#!/usr/bin/env python

from matplotlib.pyplot import plot,figure,subplots,xlim,ylim,vlines,legend,fill_between,cm

from numpy import loadtxt,arcsin,sin,pi,array,asarray,minimum,concatenate,linspace,arange
from numpy.random import randint
from glob import glob
import warnings

class Phase(dict):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def __len__(self):
        return len(self['_pd_peak_intensity'][0])

    def get_theta(self, l = [1.541874], scale = [1.0], min_theta = None, max_theta = None, min_intensity = None, first_n_peaks = None):

        if (hasattr(self, 'l_last') and hasattr(self, 'scale_last') and hasattr(self, 'min_theta_last') and
            hasattr(self, 'max_theta_last') and hasattr(self, 'min_intensity_last') and hasattr(self, 'first_n_peaks_last') and
            l == self.l_last and scale == self.scale_last and min_theta == self.min_theta_last and
            max_theta == self.max_theta_last and min_intensity == self.min_intensity_last and first_n_peaks == self.first_n_peaks_last and
            hasattr(self, 'theta') and hasattr(self, 'intensity')
        ):
            return self.theta, self.intensity
        else:
            self.l_last = l
            self.scale_last = scale
            self.min_theta_last = min_theta
            self.max_theta_last = max_theta
            self.min_intensity_last = min_intensity
            self.first_n_peaks_last = first_n_peaks

            d, i = self['_pd_peak_intensity']

            theta = []
            intensity = []

            for _l, s in zip(l,scale):
                g = _l / (2.0 * d)
                theta += [360.0 * arcsin(g) / pi]
                intensity += [i * s]

            theta = concatenate(theta)
            intensity = concatenate(intensity) / 1000.0

            mask = array([True]*len(theta))
            if min_theta:
                mask &= (theta > min_theta)
            if max_theta:
                mask &= (theta < max_theta) 
            if min_intensity:
                mask &= (intensity > min_intensity)
            self.theta, self.intensity = theta[mask], intensity[mask]

            if (self.theta.shape[0] > 0):
                if (first_n_peaks is not None):
                    self.intensity, self.theta = array(sorted(zip(self.intensity, self.theta), reverse = True)).T[:, 0:first_n_peaks]
                    self.theta, self.intensity = array(sorted(zip(self.theta, self.intensity))).T

            return self.theta, self.intensity


    def set_name(self, name):
        self['name'] = name

    def set_point(self, point):
        self['point'] = point


    def save_cif(self, filename):

        with open(filename, 'w') as file:

            if '_chemical_formula_sum' in self:
                file.write('_chemical_formula_sum  \'' + self['_chemical_formula_sum'] + '\'\n')

            if '_chemical_name_mineral' in self:
                file.write('_chemical_name_mineral  \'' + self['_chemical_name_mineral'] + '\'\n')

            if '_chemical_name_common' in self:
                file.write('_chemical_name_common  \'' + self['_chemical_name_common'] + '\'\n')

            if 'name' in self:
                file.write('name  \'' + self['name'] + '\'\n')

            if 'point' in self:
                file.write('point  ' + format(self['point'], 'd') + '\n')

            file.write('loop_\n')
            file.write('_pd_peak_d_spacing\n')
            file.write('_pd_peak_intensity\n')
            for d, i in self['_pd_peak_intensity'].T:
                d = format(d, '.6f')
                i = format(i, '.2f')
                file.write('     ' + str(d) + f'{str(i):>14}' + '\n')


    @property
    def label(self):
        if '_chemical_name_mineral' in self:
            return self['_chemical_name_mineral']
        elif '_chemical_name_common' in self:
            return self['_chemical_name_common']
        elif '_chemical_formula_sum' in self:
            return self['_chemical_formula_sum']
        else:
            return None


    def plot(self, colors = 'red', linestyles = 'dashed', label = None, lineheight = None,
         min_theta = None, max_theta = None, min_intensity = None, first_n_peaks = None, **kwargs):

        self.get_theta(min_theta = min_theta, max_theta = max_theta, min_intensity = min_intensity, first_n_peaks = first_n_peaks)

        if label is None:
            label = self.label

        if lineheight is None:
            vlines(self.theta, 0, self.intensity, colors = colors, linestyles = linestyles, label = label, **kwargs)
        else:
            vlines(self.theta, 0, lineheight, colors = colors, linestyles = linestyles, label = label, **kwargs)


class PhaseList(list):
#class PhaseList(Phase):

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

    def get_theta(self, **kwargs):
        theta = []
        intensity = []
        for phase in self:
            t, i = phase.get_theta(**kwargs)
            theta += [t]
            intensity += [i]
        return concatenate(theta), concatenate(intensity)


    def set_name(self, name):
        for phase in self:
            phase.set_name(name)

    def set_point(self, point):
        for phase in self:
            phase.set_point(point)


    def plot(self, cmap = 'tab10', min_theta = None, max_theta = None, min_intensity = None, first_n_peaks = None, **kwargs):
        cmap_sel = cm.get_cmap(cmap)
        for i, phase in enumerate(self):
            idx_color = i % cmap_sel.N
            phase.plot(min_theta = min_theta, max_theta = max_theta, min_intensity = min_intensity,
                first_n_peaks = first_n_peaks, colors = cmap_sel(idx_color), **kwargs)


    def random(self):
        idx = randint(self.__len__())
        return self[idx]


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

        return self


    def random(self):
        x = list(self.values())
        idx = randint(len(x))
        return x[idx]
