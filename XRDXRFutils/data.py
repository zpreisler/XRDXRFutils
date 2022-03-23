from scipy.optimize import curve_fit
from numpy import pi,arctan
from numpy import loadtxt,frombuffer,array,asarray,linspace,arange,trapz,zeros,empty
from scipy.interpolate import interp1d
from matplotlib.pyplot import plot,xlim,ylim,xlabel,ylabel
import os

from multiprocessing import Pool

from glob import glob
import re
import h5py
from .spectra import SyntheticSpectraXRF

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

            integral = sum((gy[1:] + gy[:-1]) * 0.5 * (_gx[1:] - _gx[:-1] ))
            iy += [integral]
            
            ay = by
            
        iy = array(iy)
                   
        return iy

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

class SyntheticDataXRF(Data):
    """
    Syntetic XRF data class
    """
    
    """
    Namespace
    """
    name = 'sxrf'
    
    def __init__(self, rl_atnum_list = None, skip_element = False):
        super().__init__()
        self.nbins = None
        self.rl_atnum_list = rl_atnum_list
        if self.rl_atnum_list:
            for i,item in enumerate(rl_atnum_list):
                if not isinstance(item, int):
                    raise TypeError(f'{item} at index {i} is not integer.\nIntegers are expected for Atomic Numbers')
        self.skip_element = skip_element
    
    def __len__(self):
        if hasattr(self, 'spe_objs'):
            return len(self.spe_objs)
        elif hasattr(self, 'data'):
            return len(self.data)
        return 0
    
    def set_nbins(self, nbins):
        self.nbins = nbins
    
    def read(self, outdata_path):
        if not self.rl_atnum_list:
            raise ValueError("Atomic numbers list required to read the data\nSet 'rl_atnum_list' attribute or initialize a new instance.")
        self.rl_atnum_list = sorted(self.rl_atnum_list)
        self.path = outdata_path
        xmso_filenames = []
        if not os.path.isdir(outdata_path):
            raise FileNotFoundError(f"No such file or directory: {outdata_path}")
        for path, dirs, files in os.walk(outdata_path):
            # to do: use glob to select xmso files
            for _file in files:
                xmso_filenames.append(os.path.join(path, _file))
        print(f"Reading SXRF data from {outdata_path}")
        self.metadata["rl_atnum_list"] = self.rl_atnum_list
        self.spe_objs = [s for s in self.__read__(xmso_filenames) if s != None]
        self.metadata["path"] = outdata_path
        
        return self
    
    def _get_labels(self, symbols, lines):
        """Generator"""
        if len(symbols) != len(lines):
            raise ValueError("Symbols and lines differ in length")
        for s in self.spe_objs:
            np_labels = zeros((len(symbols)))
            for fluo in s.fluorescence_lines:
                try:
                    sym_index = symbols.index(fluo.symbol)
                    np_labels[sym_index] = fluo.lines[lines[sym_index]]
                except ValueError:
                    pass
            yield np_labels
    
    def get_data_and_labels(self, symbols, lines):
        if not hasattr(self, 'spe_objs'):
            raise RuntimeError("xmso files not readed yet")
        self.energy = self.spe_objs[0].energy
        self.data = asarray([s.counts for s in self.spe_objs])
        self.labels = asarray([l for l in self._get_labels(symbols, lines)])
        self.metadata["symbols"] = symbols
        self.metadata["lines"] = lines
        
        return self
    
    def process_file(self, filename):
        sxrf = SyntheticSpectraXRF(self.rl_atnum_list, self.skip_element)
        self.nbins and sxrf.set_nbins(self.nbins)
        s = sxrf.from_file(filename)
        return s
    
    def __read__(self, xmso_filenames):
        if not self.rl_atnum_list:
            raise RuntimeError("missing required atomic numbers list")
        self.rl_atnum_list = sorted(self.rl_atnum_list)
        with Pool() as p:
            results = p.map(self.process_file, xmso_filenames)
        # with ThreadPoolExecutor() as executor:
            # results = executor.map(process_file, xmso_filenames)
        return results
    
    def get_sim_parameters(self, local = False):
        if not hasattr(self, 'spe_objs'):
            raise RuntimeError("xmso files not readed yet")
        len_data = len(self.spe_objs)
        if local:
            self.time = empty((len_data))
            self.weight_fractions = zeros((len_data,len(self.rl_atnum_list)))
            self.reflayer_thicknes = empty((len_data))
            self.sublayer_thicknes = empty((len_data))
            for i, s in enumerate(self.spe_objs):
                self.time[i] = s.time
                self.weight_fractions[i] = s.weight_fractions
                self.reflayer_thicknes[i] = s.reflayer_thicknes
                self.sublayer_thicknes[i] = s.sublayer_thicknes
            return
        sp = SimParameters(len_data)
        for i, s in enumerate(self.spe_objs):
            sp.time[i] = s.time
            # sp.reflayer_elements += s.reflayer_elements
            sp.weight_fractions.append(s.weight_fractions)
            sp.reflayer_thicknes[i] = s.reflayer_thicknes
            sp.sublayer_thicknes[i] = s.sublayer_thicknes
        sp.weight_fractions = asarray(sp.weight_fractions).reshape(len_data, -1)
        return sp
        
    
    def save_h5(self, filename = None):
        if not hasattr(self, 'spe_objs'):
            raise RuntimeError("xmso files not readed yet")
        if not hasattr(self, "data"):
            raise RuntimeError("Data and labels not yet genarated")
        if filename == None:
            if hasattr(self, "path"):
                filename = os.path.join(self.path, self.name + '.h5')
            else:
                filename = os.path.join(os.getcwd(), self.name + '.h5')
        if not hasattr(self,'reflayer_thicknes'):
            self.get_sim_parameters(local = True)
        self.metadata["rl_atnum_list"] = self.rl_atnum_list
        print('Saving:',filename)
        with h5py.File(filename,'w') as f:

            for k,v in self.metadata.items():
                f.attrs[k] = v

            dataset = f.create_dataset('data', data = self.data)
            dataset = f.create_dataset('labels', data = self.labels)
            dataset = f.create_dataset('reflayer_thicknes', data = self.reflayer_thicknes)
            dataset = f.create_dataset('sublayer_thicknes', data = self.sublayer_thicknes)
            # dataset = f.create_dataset('reflayer_elements', data = sp.reflayer_elements)
            dataset = f.create_dataset('weight_fractions', data = self.weight_fractions)
            dataset = f.create_dataset('energy', data = self.energy)
            dataset = f.create_dataset('time', data = self.time)

    def load_h5(self,filename):

            print('Loading:',filename)
            with h5py.File(filename,'r') as f:
                
                self.data = f['data'][:]
                self.labels = f['labels'][:]
                self.reflayer_thicknes = f['reflayer_thicknes'][:]
                self.sublayer_thicknes = f['sublayer_thicknes'][:]
                # self.reflayer_elements = f['reflayer_elements'][:]
                self.weight_fractions = f['weight_fractions'][:]
                self.energy = f['energy'][:]
                self.time = f['time'][:]

                for k,v in f.attrs.items():
                    self.metadata[k] = v

class SimParameters:
    """
    Synthetic data parameters class
    """
    def __init__(self, len_data = 1):
        self.len_data = len_data
        self.time = empty((len_data))
        self.weight_fractions = []
        self.reflayer_thicknes = empty((len_data))
        self.sublayer_thicknes = empty((len_data))

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

        print("Reading XRD data from",self.path)
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

def resample(x,y,nbins=1024,bounds=(0,30)):
    """
    Simple resample code. For debugging.
    """
    
    f = interp1d(x,y,fill_value='extrapolate')
    
    new_x = linspace(*bounds,nbins + 1)
    new_y = f(new_x)
    
    ax = new_x[0]
    ay = new_y[0]
    
    ix = []
    iy = []

    for bx,by in zip(new_x[1:],new_y[1:]):
        f = (x > ax) & (x < bx)

        gx = array([ax,*x[f],bx])
        gy = array([ay,*y[f],by])

        integral = sum((gy[1:] + gy[:-1]) * 0.5 * (gx[1:] - gx[:-1] ))

        ix += [(ax + bx) * 0.5]
        iy += [integral]

        ax = bx
        ay = by
        
    ix = array(ix)
    iy = array(iy)
               
    return ix,iy
