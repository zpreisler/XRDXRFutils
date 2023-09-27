from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from math import ceil
from numpy import (pi, arctan, loadtxt, frombuffer, array, asarray,
    linspace, arange, trapz, flip, stack, where, zeros, empty, unravel_index,
    ravel_multi_index, concatenate, append, maximum, nanmin, nanmax, rot90,
    quantile, clip, object_, uint16, flip)
from matplotlib.pyplot import plot, xlim, ylim, xlabel, ylabel
from os.path import basename, join
from os.path import dirname

from .calibration import Calibration
from .spectra import SyntheticSpectraXRF
from .utils import convolve3d, snip3d

from PIL import Image

from multiprocessing import Pool, cpu_count

from glob import glob
import re
import h5py
import warnings
import io


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
        self.check_attributes = []


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


    def remove_background(self, std_kernel = 3, window_snip = 32, offset_background = 0):
        print('Removing background...')
        self.background = snip3d(convolve3d(self.data, n = ceil(3 * std_kernel + 1), std = std_kernel), m = window_snip)
        self.offset_background = offset_background
        self.data_no_bg = self.data - self.background
        self.data_no_bg = where(self.data_no_bg > offset_background, self.data_no_bg, 0)
        self.rescaling = nanmax(self.data_no_bg, axis = 2, keepdims = True)
        self.intensity = self.data_no_bg / self.rescaling
        self.signal_background_ratio = self.data.sum(axis = 2, keepdims = True) / self.background.sum(axis = 2, keepdims = True)
        self.signal_background_ratio = maximum(self.signal_background_ratio, 0)
        print('Done.')
        return self

    def smooth_channels(self, std_kernel = 0):
        print('Smoothing along channels...')
        data_smoothed = convolve3d(self.data_no_bg, n = ceil(3 * std_kernel + 1), std = std_kernel)
        self.intensity = data_smoothed / nanmax(data_smoothed, axis = 2, keepdims = True)
        print('Done.')
        return self


    def flip(self, axis):
        for name_attr in ['data', 'labels', 'weights', 'background', 'rescaling', 'intensity', 'signal_background_ratio']:
            if hasattr(self, name_attr):
                setattr(self, name_attr, flip(getattr(self, name_attr), axis = axis))
        return self


    def rotate(self, k):
        for name_attr in ['data', 'labels', 'weights', 'background', 'rescaling', 'intensity', 'signal_background_ratio']:
            if hasattr(self, name_attr):
                setattr(self, name_attr, rot90(getattr(self, name_attr), k = k, axes = (0, 1)))
        return self


    def correct_pixels(self, indices_to_correct):
        for name_attr in ['data', 'background', 'rescaling', 'intensity', 'signal_background_ratio']:
            if hasattr(self, name_attr):
                x = getattr(self, name_attr)
                x[indices_to_correct] = x[~indices_to_correct].mean(axis = 0, keepdims = True)
                setattr(self, name_attr, x)
        print(f'{indices_to_correct.sum()} pixels out of {self.shape[0] * self.shape[1]} were corrected.')
        return self

    def correct_quantile_pixels(self, qtl):
        data_max = (self.data - self.background).max(axis = 2)
        indices_to_correct = data_max > quantile(data_max, qtl)
        return self.correct_pixels(indices_to_correct)

    def correct_specific_pixels(self, list_x_y):
        indices_to_correct = zeros(self.shape[:2], bool)
        for x, y in list_x_y:
            indices_to_correct[y, x] = True
        return self.correct_pixels(indices_to_correct)


    def save_h5(self,filename = None):

        if filename == None:
            filename = self.path + '/' + self.name + '.h5'

        print('Saving:', filename)
        with h5py.File(filename, 'w') as f:

            for k, v in self.metadata.items():
                f.attrs[k] = v

            if hasattr(self, 'data'):
                dataset = f.create_dataset('data', data = self.data, compression = 'lzf')
                dataset = f.create_dataset('x', data = self.x)

            for attr in ['labels', 'weights'] + self.check_attributes:
                if hasattr(self, attr):
                    dataset = f.create_dataset(attr, data = getattr(self, attr))
            
            if hasattr(self, 'calibration'):
                calibration = f.create_group('calibration')

                for attr in ['x', 'y', 'opt']:
                    if hasattr(self.calibration,attr):
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

            for attr in ['labels', 'weights'] + self.check_attributes:
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
        cls.metadata = self.metadata.copy()

        if hasattr(self,'labels'):
            cls.labels = self.labels
        
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

        print("Reading XRF data...")
        self.__read__(filenames)
        print("Done.")

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
    
    def read_from_map(self, path = None):
        """
        Reads XRF data from .map files.
        """
        self.path = path
        self.metadata['path'] = path
        filenames = sorted(glob(join(path,'*Z0*.map')), key = lambda x: int(re.sub('\D','',x)))
        if not filenames:
            warnings.warn('No files found')
        
        fs = basename(filenames[0]).split('_')[1]
        ii = fs.find('Z0')
        rowlen = int(fs[:ii])
        
        print("Reading XRF data...")
        self.__read_map__(filenames, rowlen = rowlen)
        print("Done.")
        
        return self
    
    def __read_map__(self, filenames, shape = (-1,2048), rowlen = None):
        
        def read_map(filename, shape = (-1,2048), rowlen = None, n = None):
            buffer = io.BytesIO()
            nchannels = shape[1]
            with open(filename, 'rb') as f:
                buffer.write(f.read())

            bl = len(buffer.getbuffer())
            hlen = bl%2048

            buffer.seek(hlen*2)
            x = frombuffer(buffer.read(), uint16)

            x = x.byteswap()

            size_idx = where(((x % nchannels) == 0) & (x>0))[0]

            newx = []
            for i in size_idx:
                for j in range(x[i]//nchannels):
                    a = i+1+(nchannels)*j
                    b = a+nchannels
                    newx += [x[a:b]]
            
            if len(newx) > rowlen: newx = newx[:rowlen]
            if n%2 == 0:
                newx = asarray(newx + [zeros(nchannels)]*(rowlen-len(newx)))
            else:
                newx = asarray([zeros(nchannels)]*(rowlen-len(newx)) + newx)

            newx = newx.reshape(*shape)

            return newx
        
        x = [read_map(filename, shape, rowlen, n) for n,filename in enumerate(filenames)]
        
        x = asarray(x)
        
        print("Flipping odd rows...")
        x[1::2] = flip(x[1::2], axis=1)
        
        self.data = x
        self._x = zeros(self.data.shape[2])
    
    def read_tiff(self, path = None):
        
        filenames = sorted(glob(path + '*.tif*'))

        labels = []
        for s in filenames:
            s = basename(s).split('.')[0]
            #s = s[s.rfind('/') + 1:]
            #s = s[:s.rfind('.')]
            labels += [s]

        self.metadata['labels'] = labels

        x = []
        for filename in filenames:
            img = Image.open(filename)
            x += [img]

        self.labels = stack(x,axis=2)

        return self

    def __select_labels(self,labels):
        
        select = []
        x = self.metadata['labels']

        for label in labels:
            w = list(where(array(x) == label)[0])
            if w:
                select += w

        self.metadata['labels'] = asarray(labels,dtype=object)
        self.labels = self.labels[...,select]

        return self

    def select_labels(self,labels):
        
        new_labels = []
        x = self.metadata['labels']

        for label in labels:
            if not label in x:
                new_labels += [label]

        print("Adding empty labels:",new_labels)

        z = zeros([self.labels.shape[0],self.labels.shape[1],len(new_labels)])

        self.labels = concatenate([self.labels,z],axis=2)
        self.metadata['labels'] = append(x,new_labels)

        self.__select_labels(labels)

        return self


    def map_correct_quantile_pixels(self, qtl):
        qtl_calculated = quantile(self.labels, qtl, axis = (0, 1), keepdims = True)
        n_corrected = (self.labels > qtl_calculated).sum() / self.labels.shape[-1]
        self.labels = clip(self.labels, None, qtl_calculated)
        print(f'{n_corrected:.0f} pixels out of {self.labels.shape[0] * self.labels.shape[1]} were corrected on average in each XRF map.')
        return self

    def map_correct_specific_pixels(self, list_x_y):
        for i in range(self.labels.shape[2]):
            min_value = nanmin(self.labels[..., i])
            for x, y in list_x_y:
                self.labels[y, x, i] = min_value
        return self

    def map_correct_scale(self):
        self.labels = clip(self.labels, 0, None)
        return self


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
        self.check_attributes += ['background', 'signal_background_ratio', 'rescaling', 'intensity']

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
        print("Done.")

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


    def generate_spatial_smooth(self, step = 2, method = 'mean'):
        if method not in ['mean', 'max']:
            raise Exception('generate_spatial_smooth: invalid method parameter.')

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

        if hasattr(self, 'calibration'):
            if hasattr(self.calibration, 'opt'):
                data_new.calibration = Calibration(data_new).from_parameters(self.calibration.opt)

        print('Done.')
        #data_new.background_elimination_and_smoothing()
        return data_new


class SyntheticDataXRF(DataXRF):
    """
    Syntetic XRF data class
    """
    
    """
    Namespace
    """
    name = 'synth_xrf'
    
    def __init__(self, nproc = None):
        
        super().__init__()
        
        #delattr(self, "calibration")
        
        self.nproc = nproc if nproc else (cpu_count() - 1)
        
        self.path = None
        
        self.name = "synth_xrf"
        
        self.layers_names = None
            
    #@property
    #def x(self):
    #    if hasattr(self, "_x"):
    #        return self._x
    
    def set_layers_names(self, names):
        self.layers_names = names
    
    def _read_xmso(self,filename):
        synt = SyntheticSpectraXRF(self.layers_names).from_file(filename)
        return synt
    
    def __read__(self):
        
        if hasattr(self, "xmso_list"):
            with Pool(processes=self.nproc) as p:
                results = p.map(self._read_xmso, self.xmso_list)
        else:
            raise RuntimeError(f"{self} has no attribute xmso_list")
        
        return results
    
#     def _expand(self, _array, index_map):
#             res = []
#             for k,v in index_map.items():
#                 if k in _array:
#                     res += [_array[k]]
#                 else:
#                     res += [0]
#             return res

    def _expand(self, datadict, _set):
            res = []
            for s in _set:
                if s in datadict:
                    res += [datadict[s]]
                else:
                    res += [0]
            return res
    
    def _get_labels(self):
        if hasattr(self,"spectra"):
            return [self._expand(s.labels, self.labels_set) for s in self.spectra]
        
    def _get_data(self):
        if hasattr(self, "spectra"):
            return [s.counts for s in self.spectra]

    def _get_wfrac(self, layer):
        if hasattr(self, "spectra"):
            return asarray([self._expand(s.layers[layer], self.elem_set) for s in self.spectra])
        elif hasattr(self, "layers"):
            out = zeros((self.__len__(), len(self.metadata['elements'])))
            for i in range(self.__len__()):
                for j,e in enumerate(self.metadata['elements']):
                    if e in self.layers[layer]['elements'][i]:
                        eidx = where(self.layers[layer]['elements'][i] == e)[0][0]
                        out[i,j] = self.layers[layer]['weight_fractions'][i,eidx]
            return out
    
    def _get_pigments(self, layer):
        if hasattr(self, "spectra"):
            return asarray([s.layers[layer].pigments for s in self.spectra])
            #return [s.layers[layer].pigments for s in self.spectra]
        elif hasattr(self, "layers"):
            return self.layers[layer]['pigments']
            #return self.layers[layer]['pigments']

    def _get_mass_volume_fractions(self, layer):
        if layer not in self.pgmset: return
        mass_fractions = zeros((len(self.spectra),len(self.pgmset[layer])))
        volume_fractions = zeros((len(self.spectra),len(self.pgmset[layer])))
        for i,s in enumerate(self.spectra):
            layerp = s.layers[layer].pigments
            layermf = s.layers[layer].mass_fractions
            layervf = s.layers[layer].volume_fractions
            for j,p in enumerate(self.pgmset[layer]):
                k = 0
                while True:
                    try:
                        k = layerp.index(p,k)
                        mass_fractions[i,j] += layermf[k]
                        volume_fractions[i,j] += layervf[k]
                        k+=1
                    except ValueError:
                        break
        return mass_fractions, volume_fractions

    def _get_volume_fractions(self, layer):
        if hasattr(self, "spectra"):
            return asarray([s.layers[layer].volume_fractions for s in self.spectra], dtype = float)
        elif hasattr(self, "layers"):
            return self.layers[layer]['volume_fractions']
    
    def _get_mass_fractions(self, layer):
        if hasattr(self, "spectra"):
            return asarray([s.layers[layer].mass_fractions for s in self.spectra], dtype = float)
        elif hasattr(self, "layers"):
            return self.layers[layer]['mass_fractions']
    
    def _get_unconv_data(self):
        if hasattr(self, "spectra"):
            return [s.unconv_counts for s in self.spectra]
    
    def _get_thickness(self):
        if hasattr(self, "spectra"):
            return asarray([[s.layers[k].thickness  for k in s.layers.keys()] for s in self.spectra])
        elif hasattr(self, "layers"):
            return asarray([l['thickness'] for l in self.layers.values()]).T 
    
    def read(self, source):
        
        if isinstance(source, str):
            self._datadir = source
            self.path = dirname(self._datadir[:-1])
            self.metadata["path"] = self.path
            self.xmso_list = glob(join(self._datadir, '*.xmso'))
        else:
            self.xmso_list = source
        
        self.spectra = self.__read__()
        
        if not self.layers_names:
            self.layers_names = self[0].layers_names
        
        self._x = self.spectra[0].energy
        
        self.labels_set = set()
        self.elem_set = set()
        self.pgmset = {}
        for s in self.spectra:
            self.labels_set.update(list(s.labels.keys()))
            for v in s.layers.values():
                self.elem_set.update(v.elements)
            for layer in self.layers_names:
                pgmlist = s.layers[layer].pigments
                if pgmlist:
                    if layer in self.pgmset:
                        self.pgmset[layer].update(pgmlist)
                    else:
                        self.pgmset[layer] = set(pgmlist)

        self.metadata['labels'] = asarray(list(self.labels_set), dtype = object_)
        self.metadata['elements'] = asarray(list(self.elem_set), dtype = object_)
        self.metadata['layers'] = asarray(self.spectra[0].layers_names, dtype = object_)
        #self.metadata['pigments'] = {k:list(v) for k,v in self.pgmset.items()}
        
        self.channels = arange(self.spectra[0].nchannels)
#         self._labmap = {l:i for i,l in enumerate(labels)}
#         self._wfmap = {e:i for i,e in enumerate(rlelem)}
        
        self.data = asarray([self._get_data()])
        self.labels = asarray([self._get_labels()])
        self.unconv_data = asarray([self._get_unconv_data()])
        #self.weight_fractions = asarray(self._get_wfrac())
        self.thickness = asarray(self._get_thickness())
#         self.reflayer_thickness = asarray([s.layers.reference_layer.thickness for s in self.spectra])
        
        return self
    
    def __getitem__(self,i):
        if hasattr(self, "spectra"):
            return self.spectra[i]
    
    def __len__(self):
        if hasattr(self, "spectra"):
            return self.spectra.__len__()
        elif hasattr(self, "data"):
            return self.data.shape[1]
        elif hasattr(self, "layers"):
            return len(self.layers[self.metadata['layers'][0]])
    
    @property
    def shape(self):
        if hasattr(self, "data"):
            return self.data.shape
        return None
    
    def save_h5(self,filename = None):

        if filename == None:
            if not self.path:
                er = 'Path attribute is not set because spectra have been read from a list\n'
                er += 'Set the path attribute or use a full filename as argument'
                raise ValueError(er)
            filename = join(self.path,f"{basename(self.path)}_{self.name}.h5")
            #filename = self.path + '/' + self.name + '.h5'

        print('Saving:',filename)
        with h5py.File(filename,'w') as f:

            for k,v in self.metadata.items():
                f.attrs[k] = v
            
            if hasattr(self, 'data'):
                dataset = f.create_dataset("data", data = self.data, compression = 'lzf')
                dataset = f.create_dataset("x", data = self._x)

            for attr in ['unconv_data','labels']:
                if hasattr(self,attr):
                    dataset = f.create_dataset(attr,data = getattr(self,attr), compression = 'lzf')

            if hasattr(self, 'spectra'):
                layers = f.create_group('layers')
                for l in self.layers_names:
                    layers.create_group(l)
                    layers[l].create_dataset('thickness', data = asarray([s.layers[l].thickness for s in self.spectra]), compression = 'lzf')
                    layers[l].create_dataset('weight_fractions', data = self._get_wfrac(l), compression = 'lzf')
                    #layers[l].create_dataset('elements', data = asarray([s.layers[l].elements for s in self.spectra]).astype('S2'))
                    if hasattr(self.spectra[0].layers, 'pigments'):
                        if l in self.pgmset:
                            layers[l].create_dataset('pigments', data = array(list(self.pgmset[l])).astype('S32'))
                            mf,vf = self._get_mass_volume_fractions(l)
                            layers[l].create_dataset('volume_fractions', data = vf)
                            layers[l].create_dataset('mass_fractions', data = mf)

            if hasattr(self, 'layers'):
                layers = f.create_group('layers')
                for l in self.layers_names:
                    layers.create_group(l)
                    for k,v in self.layers[l].items():
                        if k == 'elements':
                            layers[l].create_dataset(k, data = v.astype('S2'))
                        elif k == 'pigments':
                            layers[l].create_dataset(k, data = v.astype('S32'))
                        else:
                            layers[l].create_dataset(k, data = v)
            
        return self
    
    def save_layers(self, filename = None):
        
        if filename == None:
            if not self.path:
                er = 'Path attribute is not set because spectra have been read from a list\n'
                er += 'Aet the path attribute or use a full filename as argument'
                raise ValueError(er)
            filename = join(self.path,f"{basename(self.path)}_{self.name}_layers.h5")
        
        if hasattr(self, "spectra"):
            print('Saving:',filename)

            with h5py.File(filename,'w') as f:

                layers = f.create_group('layers')
                for l in self.layers_names:
                    layers.create_group(l)
                    layers[l].create_dataset('thickness', data = asarray([s.layers[l].thickness for s in self.spectra]))
                    layers[l].create_dataset('weight_fractions', data = self._get_wfrac(layer))
                    #layers[l].create_dataset('elements', data = asarray([s.layers[l].elements for s in self.spectra]).astype('S2'))
                    if hasattr(self.spectra[0].layers, 'pigments'):
                        layers[l].create_dataset('pigments', data = self._get_pigments(l).astype('S32'))
                        mf,vf = self_get_mass_volume_fractions(self, l)
                        layers[l].create_dataset('volume_fractions', data = vf)
                        layers[l].create_dataset('mass_fractions', data = mf)

                f.attrs['layers'] = self.layers_names
        
        return self
                
    
    def load_h5(self,filename):

        print('Loading:',filename)
        with h5py.File(filename,'r') as f:

            if 'data' in f:
                self.data = f.get('data')[()]
                self._x = f.get('x')[()]

            for attr in ['unconv_data', 'labels']:
                if attr in f:
                    setattr(self,attr,f.get(attr)[()])
            
            for k,v in f.attrs.items():
                self.metadata[k] = v
            self.path = self.metadata['path'] if "path" in self.metadata else ""

            if "/layers" in f:
                self.layers = {}
                layers = list(f['layers'].keys())
                self.layers_names = layers
                for l,data in f['layers'].items():
                    self.layers[l] = {}
                    for k,v in data.items():
                        if k == 'elements':
                            self.layers[l][k] = v[()].astype('U2')
                        elif k == 'pigments':
                            self.layers[l][k] = v[()].astype('U32')
                        else:
                            self.layers[l][k] = v[()]

        return self


    def load_layers(self, filename):
        
        print('Loading:',filename)
        with h5py.File(filename,'r') as f:
            if "/layers" in f:
                self.layers = {}
                layers = list(f['layers'].keys())
                self.layers_names = layers
                for l,data in f['layers'].items():
                    self.layers[l] = {}
                    for k,v in data.items():
                        if k == 'elements':
                            self.layers[l][k] = v[()].astype('U2')
                        elif k == 'pigments':
                            self.layers[l][k] = v[()].astype('U32')
                        else:
                            self.layers[l][k] = v[()]
     
        return self
                
    
    def resample(self, nbins = 1024, bounds = (0,30), full = False):
        
        res = super().resample(nbins, bounds)
        
        if hasattr(self, "unconv_data"):
            data = SyntheticDataXRF()
            setattr(data, "data", self.unconv_data)
            data._x = self._x
            rdata = data.resample(nbins, bounds)
            setattr(res, "unconv_data", rdata.data)
        
        res.metadata = self.metadata
        
        if full:
            if hasattr(self, "layers"):
                res.layers = self.layers
        
        return res
        

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
