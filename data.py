from scipy.optimize import curve_fit
from numpy import pi,arctan
from numpy import loadtxt,frombuffer,array,asarray,linspace

from matplotlib.pyplot import plot,xlim,ylim,xlabel,ylabel

from glob import glob
import re

class Calibration():
    """
    Calibration class
    """
    def __init__(self,fce):
        self.fce = fce

    def from_file(self,filename):

        self.x,self.y = loadtxt(filename,unpack = True, dtype = 'float')
        self.opt,opt_var = curve_fit(self.fce,self.x,self.y)

        return self

    def plot(self):
        """
        Plot the calibration function
        """
        x = linspace(0.0,self.x.max())
        y = self.fce(x,*self.opt)

        plot(x,y,'k-',lw = 1)
        plot(self.x,self.y,'.')

        xlim(0,x[-1])
        ylim(0,y[-1])

        xlabel(r'$x$')
        ylabel(r'$y$')

class Data():
    """
    Data Class
    """
    def __init__(self):
        pass

    @staticmethod
    def fce_calibration(x,a,b):
        """
        Linear calibration function
        """
        return a * x + b

    def calibrate_from_file(self,filename):
        """
        Read data from file and fit the calibration curve

        Calibration parameters are stored in self.opt

        returns: self
        """
        self.calibration = Calibration(self.fce_calibration).from_file(filename)

        return self

class DataXRF(Data):
    """
    XRF data class
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def fce_calibration(x,a,b,c):
        """
        XRF calibration function 
        """
        return a * x**2 + b * x + c

    def read(self,path=None):
        """
        Reads XRF data from .edf files.
        """
        filenames = sorted(glob(path + '/*Z0*.edf'), key=lambda x: int(re.sub('\D','',x)))

        print("Reading XRF data")
        self.__read__(filenames)
        print("Done")

        return self

    def __read__(self,filenames,n=14,shape=(-1,2048)):

        def read_edf(filename,n,shape=(-1,2048)):
            with open(filename,'rb') as f:
                for _ in range(n):
                    f.readline()
                x = frombuffer(f.read(),'d')
                x = x.reshape(*shape)

            return x

        x = [read_edf(filename,n,shape) for filename in filenames]

        self.data = asarray(x)[::-1]
        self.shape = self.data.shape

class DataXRD(Data):
    """
    XRD data class
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def fce_calibration(x,a,beta,s):
        """
        XRF calibration function 
        """
        return (arctan((x+a)/s)) * 180 / pi + beta
