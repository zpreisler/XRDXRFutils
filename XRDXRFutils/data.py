from scipy.optimize import curve_fit
from numpy import pi,arctan
from numpy import loadtxt,frombuffer,array,asarray,linspace,arange

from matplotlib.pyplot import plot,xlim,ylim,xlabel,ylabel

from glob import glob
import re
import h5py

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

        xlim(x[0],x[-1])
        ylim(y[0],y[-1])

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

    @property
    def x(self):
        return self.fce_calibration(arange(self.data.shape[-1]),*self.calibration.opt)

    def save_h5(self,filename = None):

        if filename == None:
            filename = self.path + '/' + 'data.h5'

        print('Saving:',filename)
        with h5py.File(filename,'w') as f:
            f.create_dataset('data',data = self.data)

        return self

    def load_h5(self,filename):

        print('Loading:',filename)
        with h5py.File(filename,'r') as f:
            x = f['data']
            self.data = x[:]
            self.shape = self.data.shape

        return self

    def read_params(self,filename=None):
        """
        Process the scanning parameters file.

        name: str
            name of the parameters file
        
        Returns: dictionary
        """

        print('Reading parameters from:',filename)
        self.params = {}

        with open(filename,'r') as f:
            for line in f:
                try:
                    key,value = line.split('=')
                    self.params[key] = value

                except:
                    axis = re.search('AXIS: (\S)',line)
                    value = re.search('STEP: (\d+)',line)
                    if axis and value:
                        self.params[axis.group(1)] = int(value.group(1))
        self.shape = (self.params['y'],self.params['x'],-1)

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

    def read(self,path = None):
        """
        Reads XRF data from .edf files.
        """
        self.path = path
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
    def fce_calibration(x,a,s,beta):
        """
        XRD calibration function 
        """
        return (arctan((x+a)/s)) * 180 / pi + beta

    def read(self,path = None):
        """
        Reads XRD data from .dat files.
        """
        self.path = path
        filenames = sorted(glob(self.path + '/[F,f]rame*.dat'), key=lambda x: int(re.sub('\D','',x)))

        print("Reading XRD data")
        self.__read__(filenames)
        print("Done")

        return self

    def __read__(self,filenames):

        z = []
        for i,filename in enumerate(filenames):
            """
            Strangely this is faster then loadtxt.
            """
            with open(filename,mode='r') as f:
                y = [int(line.split()[-1]) for line in f]
            z += [asarray(y)]

        z = asarray(z).reshape(self.shape)

        """
        Invert rows
        """
        for i,y in enumerate(z):
            if i % 2 == 0:
                y[:] = y[::-1]

        self.data = z
