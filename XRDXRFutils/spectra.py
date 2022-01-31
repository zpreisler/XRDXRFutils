from numpy import loadtxt,arctan,pi,arange,array
from matplotlib.pyplot import plot
from .utils import snip,convolve

class Spectra():
    def __init__(self):
        pass

    def from_array(self,x):
        self.counts = x
        self.channel = arange(self.counts.__len__())

        return self

    def from_file(self,filename):
        self.counts = loadtxt(filename,unpack=True,usecols=1)
        self.channel = arange(self.counts.__len__())

        return self

class SpectraXRF(Spectra):
    def __init__(self):
        super().__init__()

class SpectraXRD(Spectra):
    def __init__(self):
        super().__init__()

    def from_array(self,x):
        self.counts = x
        self.channel = arange(self.counts.__len__(),dtype='int')
        self.intensity = self.relative_intensity()

        return self

    def from_file(self,filename):
        self.counts = loadtxt(filename,unpack=True,dtype='int',usecols=1)
        self.channel = arange(self.counts.__len__(),dtype='int')
        self.intensity = self.relative_intensity()

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
        return self.fce_calibration(self.channel,*self.opt)

    def theta_range(self):
        x = array([self.channel[0],self.channel[-1]])
        return self.fce_calibration(x,*self.opt)

    def background(self,n=21,std=3,m=32):
        x = self.counts
        return snip(convolve(x,n=n,std=std),m=m)

    def relative_intensity(self,n=21,std=3,m=32):
        y = self.counts - self.background(n=n,std=std,m=m)
        return y / y.max()

    def plot(self,*args,**kwargs):
        plot(self.theta,self.intensity,*args,**kwargs)
