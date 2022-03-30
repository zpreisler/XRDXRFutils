from numpy import loadtxt,arctan,pi,arange,array
from matplotlib.pyplot import plot
from .utils import snip,convolve
from .data import Calibration

class Spectra():
    def __init__(self):
        self.calibration = Calibration(self)

    def from_array(self,x):
        self.counts = x
        self.channel = arange(self.counts.__len__())

        return self

    def from_file(self,filename):
        self.counts = loadtxt(filename,unpack=True,usecols=1)
        self.channel = arange(self.counts.__len__())

        return self

    def from_Data(self,data,x=0,y=0):
        self.counts = data.data[x,y]
        self.channel = arange(self.counts.__len__(),dtype='int')

        return self

class SpectraXRF(Spectra):
    def __init__(self):
        super().__init__()

class SpectraXRD(Spectra):
    def __init__(self):
        super().__init__()

    def from_array(self, x):
        self.counts = x
        self.channel = arange(self.counts.__len__(), dtype = 'int')

        #self.calculate_signals()
        return self

    def from_file(self, filename):

        counts = loadtxt(filename, unpack = True, dtype = 'int', usecols = 1)
        return self.from_array(counts)

    def from_Data(self, data, x = 0, y = 0):

        self.calibrate_from_parameters(data.opt)

        self.counts = data.data[x, y]
        self.rescaling = data.rescaling[x, y]
        self.intensity = data.intensity[x, y]

        self.channel = arange(self.counts.__len__(), dtype = 'int')

        return self#.from_array(counts)

    #def calculate_signals(self, n = 21, std = 3, m = 32):
    def remove_background(self, n = 21, std = 3, m = 32):

        background = snip(convolve(self.counts, n = n, std = std), m = m)
        #self.counts_clean = self.counts - self.background
        counts = self.counts - background

        self.rescaling = counts.max()
        self.intensity = counts / self.rescaling

        return self

    def calibrate_from_parameters(self, opt):

        self.calibration.from_parameters(opt)
        return self

    def calibrate_from_file(self, filename):
        """
        Read data from file and fit the calibration curve

        Calibration parameters are stored in self.opt

        returns: self
        """
        self.calibration.from_file(filename)
        return self

    @staticmethod
    def fce_calibration(x,a,s,beta):
        """
        XRD calibration function 
            x is a channel
        """
        return (arctan((x + a) / s)) * 180 / pi + beta

    @property
    def theta(self):
        return self.fce_calibration(self.channel, *self.opt)

    def theta_range(self):
        x = array([self.channel[0], self.channel[-1]])
        return self.fce_calibration(x, *self.opt)

    def plot(self,*args,**kwargs):
        plot(self.theta,self.intensity,*args,**kwargs)

class FastSpectraXRD():
    def __init__(self):
        pass

    @staticmethod
    def fce_calibration(x,a,s,beta):
        """
        XRD calibration function 
            x is a channel
        """
        return (arctan((x + a) / s)) * 180 / pi + beta

    @property
    def theta(self):
        #return self.fce_calibration(self.channel, *self.opt)
        return self.fce_calibration(arange(1280), *self.opt)

    def theta_range(self):
        x = array([self.channel[0], self.channel[-1]])
        return self.fce_calibration(x, *self.opt)

    def plot(self,*args,**kwargs):
        plot(self.theta,self.intensity,*args,**kwargs)


