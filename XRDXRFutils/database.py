#!/usr/bin/env python

from matplotlib.pyplot import plot,figure,subplots,xlim,ylim,vlines,legend,fill_between

from numpy import loadtxt,arcsin,pi,array,asarray,minimum,concatenate
from numpy.random import randint
from glob import glob

class Phase(dict):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def __len__(self):
        return len(self['_pd_peak_intensity'][0])

    def get_theta(self,l=[1.541],scale=[1.0],max_theta=None,min_intensity=None):

        d,i = self['_pd_peak_intensity']

        theta = []
        intensity = []

        for _l,s in zip(l,scale):
            g = _l / (2.0 * d)
            theta += [360.0 * arcsin(g) / pi]
            intensity += [i * s]

        theta = concatenate(theta)
        intensity = concatenate(intensity) / 1000.0
        
        self.theta,self.intensity = array(sorted(zip(theta,intensity))).T

        f = array([True]*len(self.theta))
        if max_theta:
            f &= (self.theta < max_theta) 
        if min_intensity:
            f &= self.intensity > min_intensity

        return self.theta[f],self.intensity[f]

    def plot(self, colors='k', linestyles='dashed', label=None, **kwargs):

        if not hasattr(self,'theta'):
            self.get_theta()

        if label is None:
            label = self['_chemical_name_mineral']

        vlines(self.theta,0,self.intensity, colors=colors, linestyles=linestyles, label=label, **kwargs)

        return self

class PhaseList(list):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.label = None

        if 'label' in kwargs:
            self.label = kwargs.pop('label')

    def get_theta(self,**kwargs):
        
        theta = []
        intensity = []
        
        for phase in self:
            x,y = phase.get_theta(**kwargs)
            theta += [x]
            intensity += [y]
            
        self.theta,self.intensity = concatenate(theta),concatenate(intensity)

        return self.theta,self.intensity

    def random(self):
        idx = randint(self.__len__())
        return self[idx]

class DatabaseXRD(dict):

    def read_cifs(self,path):
        filenames = sorted(glob(path + '/*.cif'))

        i = 0
        for filename in filenames:
            phase = Phase(name=filename)

            with open(filename,'r') as f:
                for line in f:
                    x = line.split()
                    if x:
                        y = x[0]
                        if y == '_chemical_formula_sum':
                            phase[y] = ' '.join(x[1:]).replace("'",'')

                        if y == '_chemical_name_mineral':
                            phase[y] = ' '.join(x[1:]).replace("'",'')

                        if y == '_chemical_name_common':
                            phase[y] = x[1:]

                        if y == '_pd_peak_intensity':
                            z = loadtxt(f,unpack=True,dtype=float)
                            phase[y] = z

            formula = phase['_chemical_formula_sum']

            if '_chemical_name_mineral' in phase:
                key = phase['_chemical_name_mineral']
            else:
                key = formula

            if key in self:
                self[key] += PhaseList([phase])
            else:
                self[key] = PhaseList([phase],label = i)
                i += 1

        return self

    def random(self):
        x = list(self.values())
        idx = randint(len(x))
        return x[idx]