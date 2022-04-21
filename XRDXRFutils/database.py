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

    def get_theta(self, l=[1.541874], scale=[1.0], min_theta=None, max_theta=None, min_intensity=None, first_n_peaks = None):

        #FIXME
        #Recalculate when conditions are not the same

        if hasattr(self,'theta') and hasattr(self,'intensity'):
            return self.theta,self.intensity

        d, i = self['_pd_peak_intensity']

        theta = []
        intensity = []

        for _l, s in zip(l,scale):
            g = _l / (2.0 * d)
            theta += [360.0 * arcsin(g) / pi]
            intensity += [i * s]

        theta = concatenate(theta)
        intensity = concatenate(intensity) / 1000.0

        f = array([True]*len(theta))
        if min_theta:
            f &= (theta > min_theta)
        if max_theta:
            f &= (theta < max_theta) 
        if min_intensity:
            f &= (intensity > min_intensity)
        self.theta, self.intensity = theta[f], intensity[f]

        if (self.theta.shape[0] > 0):
            if (first_n_peaks is not None):
                self.intensity, self.theta = array(sorted(zip(self.intensity, self.theta), reverse = True)).T[:, 0:first_n_peaks]
                self.theta, self.intensity = array(sorted(zip(self.theta, self.intensity))).T

        return self.theta, self.intensity


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


    def plot(self, colors = 'k', linestyles = 'dashed', label = None, lineheight = None, **kwargs):

        if not hasattr(self, 'theta'):
            self.get_theta()

        if label is None:
            try:
                label = self['_chemical_name_mineral']
            except:
                label = self['_chemical_formula_sum']

        if lineheight is None:
            vlines(self.theta, 0, self.intensity, colors = colors, linestyles = linestyles, label = label, **kwargs)
        else:
            vlines(self.theta, 0, lineheight, colors = colors, linestyles = linestyles, label = label, **kwargs)

class PhaseList(list):
#class PhaseList(Phase):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if 'label' in kwargs:
            self.label_arg = kwargs.pop('label')

    @property
    def label(self):
        if hasattr(self, 'label_arg'):
            return self.label_arg
        else:
            return '[' + ', '.join([elem.label for elem in self]) + ']'

    def get_theta(self,**kwargs):

        #FIXME
        if hasattr(self,'theta') and hasattr(self,'intensity'):
            return self.theta,self.intensity
        
        theta = []
        intensity = []
        
        for phase in self:
            x,y = phase.get_theta(**kwargs)
            theta += [x]
            intensity += [y]
            
        self.theta,self.intensity = concatenate(theta),concatenate(intensity)

        return self.theta,self.intensity


    def plot(self, cmap = 'tab10', **kwargs):
        cmap_sel = cm.get_cmap(cmap)
        for i, phase in enumerate(self):
            idx_color = i % cmap_sel.N
            phase.plot(colors = cmap_sel(idx_color), **kwargs)


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
                        if y == '_chemical_formula_sum':
                            phase[y] = ' '.join(x[1:]).replace("'", '')

                        if y == '_chemical_name_mineral':
                            phase[y] = ' '.join(x[1:]).replace("'", '')

                        if y == '_chemical_name_common':
                            phase[y] = ' '.join(x[1:]).replace("'", '')

                        if y == 'name':
                            phase[y] = ' '.join(x[1:]).replace("'", '')

                        if y == 'point':
                            phase[y] = int(x[1])

                        if y == '_pd_peak_intensity':
                            z = loadtxt(f, unpack = True, dtype = float)
                            phase[y] = z

            formula = phase['_chemical_formula_sum']

            if '_chemical_name_mineral' in phase:
                key = phase['_chemical_name_mineral']
            else:
                key = formula

            phase.label = key
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
