from scipy.optimize import curve_fit
from numpy import (pi, arctan, loadtxt, frombuffer, array, asarray,
    linspace, arange, trapz, flip, stack, where, zeros, empty, unravel_index,
    ravel_multi_index)
from scipy.interpolate import interp1d
from matplotlib.pyplot import plot, xlim, ylim, xlabel, ylabel
import os

from .calibration import Calibration
from .spectra import SyntheticSpectraXRF
from .utils import convolve3d,snip3d

from PIL import Image

from multiprocessing import Pool

from glob import glob
import re
import h5py
import warnings

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
        
        #f = interp1d(self.x,self.y,fill_value='extrapolate')
        f = interp1d(self.x,self.y,fill_value = 0.0, bounds_error=False)
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
        self.check_attributes = ['background', 'signal_background_ratio', 'rescaling', 'intensity']

    @staticmethod
    def fce_calibration(x,a,b):
        """
        Linear calibration function
        """
        return a * x + b

    def calibrate_from_parameters(self, opt):
        self.calibration = Calibration(self).from_parameters(opt)
        return self

    def calibrate_from_file(self, filename):
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

    def get_x_y(self, i):
        y, x = unravel_index(i, self.shape[:2])
        return x, y

    def get_index(self, x, y):
        return ravel_multi_index((y, x), self.shape[:2])

    @property
    def x(self):
        if hasattr(self, 'calibration'):
            return self.fce_calibration(arange(self.shape[-1]), *self.calibration.opt)
        else:
            return self._x

    def remove_background(self, n = 21, std = 3, m = 32):

        print('Removing background...')
        self.background = snip3d(convolve3d(self.data, n = n, std = std), m = m)
        data = self.data - self.background

        self.signal_background_ratio = self.data.sum(axis = 2, keepdims = True) / self.background.sum(axis = 2, keepdims = True)
        self.rescaling = data.max(axis = 2, keepdims = True)
        self.intensity = data / self.rescaling

        print('Done.')
        return self

    def save_h5(self,filename = None):

        if filename == None:
            filename = self.path + '/' + self.name + '.h5'

        print('Saving:', filename)
        with h5py.File(filename, 'w') as f:

            for k, v in self.metadata.items():
                f.attrs[k] = v

            if hasattr(self, 'data'):
                dataset = f.create_dataset('data', data = self.data)
                dataset = f.create_dataset('x', data = self.x)

            for attr in ['labels', 'weights', 'background', 'signal_background_ratio', 'rescaling', 'intensity']:
                if hasattr(self, attr):
                    dataset = f.create_dataset(attr, data = getattr(self, attr))
            
            if hasattr(self, 'calibration'):
                calibration = f.create_group('calibration')

                for attr in ['x', 'y', 'opt']:
                    calibration.create_dataset(attr, data = getattr(self.calibration, attr))
                for k, v in self.calibration.metadata.items():
                    calibration.attrs[k] = v

        return self

    def load_h5(self,filename):

        print('Loading:', filename)
        with h5py.File(filename, 'r') as f:

            if 'data' in f:
                self.data = f.get('data')[()]
                self._x = f.get('x')[()]

            for attr in ['labels', 'weights', 'background', 'signal_background_ratio', 'rescaling', 'intensity']:
                if attr in f:
                    setattr(self, attr, f.get(attr)[()])

            for k, v in f.attrs.items():
                self.metadata[k] = v

            if 'calibration' in f:
                x = f['calibration']
                self.calibration = Calibration(self)
                for k, v in x.items():
                    setattr(self.calibration, k, v[:])
                for k, v in x.attrs.items():
                    self.calibration.metadata[k] = v
                self.opt = self.calibration.opt

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

        #self.data = asarray(x)[::-1]
        self.data = asarray(x)

    def read_tiff(self, path = None):
        
        filenames = sorted(glob(path + '*.tif*'))

        labels = []
        for s in filenames:
            s = s[s.rfind('/') + 1:]
            s = s[:s.rfind('.')]
            labels += [s]

        self.metadata['labels'] = labels

        x = []
        for filename in filenames:
            img = Image.open(filename)
            x += [img]

        self.labels = stack(x,axis=2)

        return self

    def select_labels(self,labels):
        
        select = []
        x = self.metadata['labels']

        for label in labels:
            w = list(where(x == label)[0])
            if w:
                select += w

        self.metadata['labels'] = asarray(labels,dtype=object)
        self.labels = self.labels[...,select]

        return self

class SyntheticDataXRF(Data):
    """
    Syntetic XRF data class
    """
    
    """
    Namespace
    """
    name = 'synth_xrf'
    
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
        """
        Generator
        """
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

        self.data = asarray([s.counts for s in self.spe_objs])
        self.energy = self.spe_objs[0].energy

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
        return results
    
    def get_sim_parameters(self, local = False):

        if not hasattr(self, 'spe_objs'):
            raise RuntimeError("xmso files not readed yet")
        len_data = len(self.spe_objs)

        if local:
            self.time = empty((len_data))
            self.weight_fractions = zeros((len_data,len(self.rl_atnum_list)))
            self.reflayer_thickness = empty((len_data))
            self.sublayer_thickness = empty((len_data))

            for i, s in enumerate(self.spe_objs):
                self.time[i] = s.time
                self.weight_fractions[i] = s.weight_fractions
                self.reflayer_thickness[i] = s.reflayer_thickness
                self.sublayer_thickness[i] = s.sublayer_thickness
            return

        sp = SimulationParams(len_data)

        for i, s in enumerate(self.spe_objs):
            sp.time[i] = s.time
            # sp.reflayer_elements += s.reflayer_elements
            sp.weight_fractions.append(s.weight_fractions)
            sp.reflayer_thickness[i] = s.reflayer_thickness
            sp.sublayer_thickness[i] = s.sublayer_thickness

        sp.weight_fractions = asarray(sp.weight_fractions).reshape(len_data, -1)

        return sp
        
    
    def save_h5(self, filename = None):

        if filename == None:
            filename = self.path + '/' + self.name + '.h5'

        print('Saving:',filename)
        with h5py.File(filename,'w') as f:

            for k,v in self.metadata.items():
                f.attrs[k] = v

            if hasattr(self,'data'):
                dataset = f.create_dataset('data',data = self.data)
                dataset = f.create_dataset('x',data = self.x)

            if hasattr(self,'labels'):
                dataset = f.create_dataset('labels',data = self.labels)

            for attr in ['reflayer_thickness','sublayer_thickness','weight_fractions','time','energy']:
                if hasattr(self,attr):
                    dataset = f.create_dataset(attr,data = getattr(self,attr))
            
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

            if 'data' in f:
                self.data = f.get('data')[()]

                if 'x' in f:
                    self._x = f.get('x')[()]
                else:
                    self._x = f.get('energy')[()]

            if 'labels' in f:
                self.labels = f.get('labels')[()]

            for attr in ['reflayer_thickness','sublayer_thickness','weight_fractions','time','energy']:
                if attr in f:
                    setattr(self,attr,f.get(attr)[()])

            for k,v in f.attrs.items():
                self.metadata[k] = v

        return self

class SimulationParams:
    """
    Synthetic data parameters class
    """
    def __init__(self, len_data = 1):

        self.len_data = len_data
        self.time = empty((len_data))

        self.weight_fractions = []
        self.reflayer_thickness = empty((len_data))
        self.sublayer_thickness = empty((len_data))

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

        self.data = z[::-1,::-1]


    def generate_smooth(self, step = 2, method = 'mean'):
        if method not in ['mean', 'max']:
            raise Exception('Invalid method parameter')

        print('Generating smooth data...')
        data_new = DataXRD()

        data_new.metadata = self.metadata.copy()
        data_new.smooth_step = step

        data_new.data = empty(self.shape)
        for i in range(0, self.shape[0], step):
            step_i = min(step, self.shape[0] - i)
            for j in range(0, self.shape[1], step):
                step_j = min(step, self.shape[1] - j)
                if method == 'mean':
                    aggr = self.data[i : (i + step_i), j : (j + step_j), :].mean(axis = (0, 1))
                else:
                    aggr = self.data[i : (i + step_i), j : (j + step_j), :].max(axis = (0, 1))
                for i_small in range(0, step_i):
                    for j_small in range(0, step_j):
                        data_new.data[i + i_small, j + j_small] = aggr

        data_new.remove_background()

        if hasattr(self, 'calibration'):
            if hasattr(self.calibration, 'opt'):
                data_new.calibration = Calibration(data_new).from_parameters(self.calibration.opt)

        print('Done.')
        return data_new


def resample(x,y,nbins=1024,bounds=(0,30)):
    """
    Simple resample code. For debugging.
    """
    s = x < bounds[-1]
    x = x[s]
    y = y[s]

    #f = interp1d(x,y,fill_value='extrapolate')
    f = interp1d(x,y,fill_value = 0.0, bounds_error=False)
    
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
