from scipy.optimize import curve_fit
from numpy import pi, arctan
from numpy import loadtxt, frombuffer, array, asarray, linspace, arange, trapz, flip
from scipy.interpolate import interp1d
from matplotlib.pyplot import plot, xlim, ylim, xlabel, ylabel

from multiprocessing import Pool

from glob import glob
import re
import h5py
import warnings

class Calibration():
    """
    Calibration class
    """
    def __init__(self,parent):

        self.metadata = {}
        self.parent = parent

        self.fce = parent.fce_calibration

    def from_file(self,filename):

        self.metadata['filename'] = filename

        self.x,self.y = loadtxt(filename,unpack = True, dtype = 'float')
        self.opt,opt_var = curve_fit(self.fce,self.x,self.y)

        self.parent.opt = self.opt.copy()

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

class Container():
    """
    Container to pass parameters to the Pool.
    """
    def __init__(self,y,x,new_x,fx,gx):

        self.y = y
        self.x = x
        self.new_x = new_x
        self.fx = fx
        self.gx =gx
    
    def resample_y(self):
        
        f = interp1d(self.x,self.y,fill_value='extrapolate')
        new_y = f(self.new_x)

        iy = []
        ay = new_y[0]
        for f,_gx,by in zip(self.fx,self.gx,new_y[1:]):

            gy = array([ay,*self.y[f],by])

            integral = sum((gy[1:] + gy[:-1]) * (_gx[1:] - _gx[:-1] ))
            iy += [integral]
            
            ay = by
            
        iy = array(iy) * 0.5

        s = sum(iy)
        if s > 0:
            scale = sum(self.y) / s
        else:
            scale = 1

        return iy * scale

class Data():
    """
    Data Class
    """

    """
    Namespace
    """
    name = 'data'

    def __init__(self):
        self.metadata = {}

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
        self.calibration = Calibration(self).from_file(filename)

        return self

    @property
    def shape(self):
        return self.data.shape

    @property
    def x(self):
        if hasattr(self,'calibration'):
            return self.fce_calibration(arange(self.shape[-1]),*self.calibration.opt)
        else:
            return self._x

    def save_h5(self,filename = None):

        if filename == None:
            filename = self.path + '/' + self.name + '.h5'

        print('Saving:',filename)
        with h5py.File(filename,'w') as f:

            for k,v in self.metadata.items():
                f.attrs[k] = v

            dataset = f.create_dataset('data',data = self.data)
            dataset = f.create_dataset('x',data = self.x)
            
            if hasattr(self,'calibration'):
                calibration = f.create_group('calibration')

                for attr in ['x','y','opt']:
                    calibration.create_dataset(attr,data = getattr(self.calibration,attr))
                for k,v in self.calibration.metadata.items():
                    calibration.attrs[k] = v

        return self

    def load_h5(self,filename):

        print('Loading:',filename)
        with h5py.File(filename,'r') as f:

            x = f['data']
            self.data = x[:]

            if 'x' in f:
                self._x = f['x'][:]

            for k,v in f.attrs.items():
                self.metadata[k] = v

            if 'calibration' in f:
                x = f['calibration']
                self.calibration = Calibration(self)
                for k,v in x.items():
                    setattr(self.calibration,k,v[:])

                for k,v in x.attrs.items():
                    self.calibration.metadata[k] = v

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

        if filename == None:

            self.params['x'] = 1
            self.params['y'] = 1
            self.params['shape'] = (1,1,-1)

            return self

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

        self.params['shape'] = (self.params['y'],self.params['x'],-1)

        return self

    @staticmethod
    def f_resample_y(x):
        """
        For Pool
        """
        return x.resample_y()

    def resample(self,nbins=1024,bounds=(0,30)):

        x,c = self.__resample_x(nbins,bounds)
        with Pool() as p:
            results = p.map(self.f_resample_y,c) 

        cls = self.__class__()
        cls.data = asarray(results).reshape(self.shape[0],self.shape[1],-1)
        cls._x = x
        
        return cls

    def __resample_x(self,nbins,bounds):
        """
        Resample the whole dataset
        """

        def resample_x(x,nbins,bounds):
            
            new_x = linspace(*bounds,nbins + 1)
            
            fx = []
            gx = []

            ax = new_x[0]
            
            for bx in new_x[1:]:
                
                f = (x > ax) & (x < bx)
                
                gx += [array([ax,*x[f],bx])]
                fx += [f]

                ax = bx
                       
            return new_x,fx,gx

        x = self.x
        y = self.data.reshape(-1,self.data.shape[-1])

        s = x < bounds[-1]
        x = x[s]
        y = y[:,s]

        new_x,fx,gx = resample_x(x,nbins,bounds)
        ix = (new_x[:-1] + new_x[1:]) * 0.5

        return ix,[Container(_y,x,new_x,fx,gx) for _y in y]

class DataXRF(Data):
    """
    XRF data class
    """

    """
    Namespace
    """
    name = 'xrf'

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
        self.metadata['path'] = path
        filenames = sorted(glob(path + '/*Z0*.edf'), key=lambda x: int(re.sub('\D','',x)))
        if not filenames:
            warnings.warn('No files found')

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

class DataXRD(Data):
    """
    XRD data class
    """

    """
    Namespace
    """
    name = 'xrd'

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
        self.metadata['path'] = path
        filenames = sorted(glob(self.path + '/[F,f]rame*.dat'), key=lambda x: int(re.sub('\D','',x)))
        if not filenames:
            warnings.warn('No files found')

        print("Reading XRD data from",self.path)
        self.__read__(filenames)
        print("Done")

        return self

    def __read__(self,filenames):

        z = []
        for i,filename in enumerate(filenames):
            """
            Strangely this is faster then loadtxt. But only reads int!
            """
            with open(filename,mode='r') as f:
                y = [int(line.split()[-1]) for line in f]
            z += [asarray(y)]

        z = asarray(z).reshape(self.params['shape'])

        """
        Invert rows
        """
        for i,y in enumerate(z):
            if i % 2 == 0:
                y[:] = y[::-1]

        self.data = flip(z, axis = 0)

def resample(x,y,nbins=1024,bounds=(0,30)):
    """
    Simple resample code. For debugging.
    """
    s = x < bounds[-1]
    x = x[s]
    y = y[s]

    f = interp1d(x,y,fill_value='extrapolate')
    
    new_x = linspace(*bounds,nbins + 1)
    new_y = f(new_x)
    
    ax = new_x[0]
    ay = new_y[0]
    
    ix = []
    iy = []

    for bx,by in zip(new_x[1:],new_y[1:]):
        f = (x >= ax) & (x < bx)

        gx = array([ax,*x[f],bx])
        gy = array([ay,*y[f],by])

        integral = sum((gy[1:] + gy[:-1]) * (gx[1:] - gx[:-1] ))

        ix += [(ax + bx) * 0.5]
        iy += [integral * 0.5]

        ax = bx
        ay = by

        
    ix = array(ix)
    iy = array(iy)

    scale = sum(y) / sum(iy)
               
    return ix,iy * scale
