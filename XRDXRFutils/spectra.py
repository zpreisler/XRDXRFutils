from numpy import loadtxt, arctan, pi, arange, array, asarray, linspace, zeros, maximum, nanmax, where, empty, empty_like
from math import ceil
from matplotlib.pyplot import plot
import xml.etree.ElementTree as et
from scipy.interpolate import interp1d
from re import sub as re_sub
import warnings

from .utils import snip, convolve
from .calibration import Calibration
from .Xmendeleev import Xmendeleev

from collections import UserDict
import yaml

class Spectra():
    def __init__(self):
        self.calibration = Calibration(self)

    def from_array(self, counts):
        self.counts = counts
        self.channel = arange(len(self.counts), dtype = 'int')
        return self

    def from_file(self, filename):
        counts = loadtxt(filename, unpack = True, dtype = 'int', usecols = 1)
        return self.from_array(counts)

    def from_Dataf(self, data, i):
        counts = data.data.reshape(-1, data.shape[2])[i]
        return self.from_array(counts)

    def from_Data(self, data, x, y):
        return self.from_Dataf(data, data.get_index(x, y))


    def remove_background(self, std_kernel = 3, window_snip = 32, offset_background = 0):
        self.background = snip(convolve(self.counts, n = ceil(3 * std_kernel + 1), std = std_kernel), m = window_snip)
        self.offset_background = offset_background
        self.counts_no_bg = self.counts - self.background
        self.counts_no_bg = where(self.counts_no_bg > offset_background, self.counts_no_bg, 0)
        self.rescaling = nanmax(self.counts_no_bg)
        return self

    def smooth_channels(self, std_kernel = 0):
        self.counts_smoothed = convolve(self.counts_no_bg, n = ceil(3 * std_kernel + 1), std = std_kernel)
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
    def fce_calibration(x, a, b, c):
        """
        Calibration function 
            x is a channel
        """
        return a * x**2 + b * x + c


    @property
    def length(self):
        return len(self.counts)


class SpectraXRF(Spectra):
    def from_spe(self, filename):
        is_found = False

        with open(filename, 'r') as o_file_xrf:
            while not is_found:
                line = o_file_xrf.readline()
                if line == '$DATA:\n':
                    is_found = True
                if not line:
                    break
            if is_found:
                line = o_file_xrf.readline()
                lines = o_file_xrf.readlines()
                counts = asarray([int(n) for l in lines for n in re_sub(' +', ' ', l).split()])
                return self.from_array(counts)
            else:
                warnings.warn(f'Unknown data format in file \'{filename}\'')
                return None

    @property
    def energy(self):
        return self.fce_calibration(self.channel, *self.opt)

    def energy_range(self):
        x = array([self.channel[0], self.channel[-1]])
        return self.fce_calibration(x, *self.opt)


class SpectraXRD(Spectra):

    def __init__(self, downsample_max = 0):
        super().__init__()
        # Downsample levels: from 0 (original sampling) to downsample_max
        # The n. of channels is divided by 2 ^ downsample_level
        if downsample_max >= 0:
            self.downsample_max = downsample_max
        else:
            self.downsample_max = 0
        self.downsample_level = 0


    def calculate_downsampled_channel(self):
        self.channel_downsampled = []
        for i in range(self.downsample_max + 1):
            self.channel_downsampled.append(arange((2**i - 1) / 2, len(self.counts), 2**i))
        return self

    def calculate_downsampled_intensity(self, intensity):
        self.intensity_downsampled = [intensity]
        for i in range(self.downsample_max):
            self.intensity_downsampled.append(0.5 * (self.intensity_downsampled[i][::2] + self.intensity_downsampled[i][1::2]))
        return self

    def downsample(self, level):
        if level > self.downsample_max:
            self.downsample_level = self.downsample_max
        elif level < 0:
            self.downsample_level = 0
        else:
            self.downsample_level = level
        return self


    def from_array(self, counts):
        self.counts = counts
        self.calculate_downsampled_channel()
        return self

    def from_components(self, opt, counts, rescaling, intensity):
        self.calibrate_from_parameters(opt)
        self.from_array(counts)
        self.rescaling = rescaling
        self.calculate_downsampled_intensity(intensity)
        return self

    def from_Dataf(self, data, i):
        return self.from_components(
            opt = data.opt.copy(),
            counts = data.data.reshape(-1, data.shape[2])[i],
            rescaling = data.rescaling.flatten()[i],
            intensity = data.intensity.reshape(-1, data.shape[2])[i]
        )


    def remove_background(self, std_kernel = 3, window_snip = 32, offset_background = 0):
        super().remove_background(std_kernel, window_snip, offset_background)
        self.calculate_downsampled_intensity(self.counts_no_bg / self.rescaling)
        return self

    def smooth_channels(self, std_kernel = 0):
        super().smooth_channels(std_kernel)
        self.calculate_downsampled_intensity(self.counts_smoothed / nanmax(self.counts_smoothed))
        return self


    @property
    def channel(self):
        return self.channel_downsampled[self.downsample_level]

    @property
    def intensity(self):
        return self.intensity_downsampled[self.downsample_level]

    @property
    def theta(self):
        return self.fce_calibration(self.channel, *self.opt)

    def theta_range(self):
        x = array([self.channel[0], self.channel[-1]])
        return self.fce_calibration(x, *self.opt)

    @staticmethod
    def fce_calibration(x, a, s, beta):
        """
        XRD calibration function 
            x is a channel
        """
        return (arctan((x + a) / s)) * 180 / pi + beta


    def plot(self, *args, **kwargs):
        plot(self.theta, self.intensity, *args, **kwargs)


class FastSpectraXRD(SpectraXRD):
    def __init__(self):
        super().__init__(downsample_max = 3)

        
xm = Xmendeleev()

def pretty(d, tab = "", buffer = ""):
    _len = len(d)
    link = "│"
    for i,(k,v) in enumerate(d.items()):
        if i == (_len - 1):
            node = "└──"
            link = " "
        else:
            node = "├──"
        if isinstance(v, UserDict) or isinstance(v, dict):
            if hasattr(v, "density"):
                k  = f'{k} density = {v.density}'
            if hasattr(v, "thickness"):
                k = f"{k} thickness = {v.thickness*1.0e4} \u03bcm"
            buffer += f'{tab}{node} {k}\n'
            #print(f'{link}{node} {k}')
            buffer = pretty(v, link + " "*3 + tab , buffer)
        else:
            buffer += f'{tab}{node} {k}  {v}\n'
            #print(f'{link}{node} {k}  {v}')
    return buffer

class Layers(UserDict):
    
    class Layer(UserDict):
        
        def __init__(self,x = None):
            super().__init__()
            
            if x:
                for element in x.findall('element'):
                    an = int(element.find('atomic_number').text)
                    self[xm.get_element(an).symbol] = float(element.find('weight_fraction').text)

                self.density = float(x.find('density').text)
                self.thickness = float(x.find('thickness').text)
        
        @property
        def elements(self):
            return list(self.keys())
        
        @property
        def weight_fractions(self):
            return asarray(list(self.values()))
        
        def __repr__(self):
            return f"""
    
    Layer(elements = {self.elements}
          weight_fractions = {self.weight_fractions}
          density = {self.density}
          thickness = {self.thickness})"""
    
    def __init__(self,xml_data = None,keys = None):
        
        super().__init__()
        
        if xml_data:
            layers = xml_data.findall('./xmimsim-input/composition/layer')

            comments = xml_data.find('./xmimsim-input/general/comments')
            
            if comments != None:
                self.comments = yaml.load(comments.text, yaml.SafeLoader)
                if not isinstance(self.comments, dict):
                    delattr(self,'comments')
            if not keys:
                if hasattr(self, "comments") and isinstance(self.comments, dict):
                    keys = list(self.comments['layers'].keys())
                else:
                    keys = [f'layer{i}' for i in range(len(layers))]

            for k,layer in zip(keys, layers):
                self[k] = self.Layer(layer)
                if hasattr(self, "comments"):
                    self[k].pigments = self.comments['layers'][k]['pigments']
                    self[k].volume_fractions = self.comments['layers'][k]['volume_fractions']
                    self[k].mass_fractions = self.comments['layers'][k]['mass_fractions']
    #         super().__init__(self.Layer(x) for x in xml_data.findall('./xmimsim-input/composition/layer'))
            rl_index = int(xml_data.find('./xmimsim-input/composition/reference_layer').text)-1
            self.reflayer_name = keys[rl_index]
        
    @property    
    def reference_layer(self):
        if hasattr(self, "reflayer_name"):
            return self[self.reflayer_name]
    
    def __str__(self):
        ret =  "Layers\n" + pretty(self)
        if hasattr(self, "comments"):
            ret += 'Layers\n' + pretty(self.comments['layers'])
        return ret
    
    @property
    def thickness(self):
        return [self[layer].thickness for layer in self]
    
    @property
    def density(self):
        return [self[layer].density for layer in self]
    
    @property
    def elements(self):
        return [self[layer].elements for layer in self]
    
    @property
    def pigments(self):
        if hasattr(self, "comments"):
            return [self[layer].pigments for layer in self]
    
            
class Labels(UserDict):
        
    def __init__(self, xml_data):
        
        super().__init__()
        flc = xml_data.findall(".//fluorescence_line_counts")
        
        self['time'] = float(xml_data.find('./xmimsim-input/detector/live_time').text)
        
        for element in flc:
            symbol = element.attrib['symbol']
            #lines = ["KL","KM", "KN", "KO","L1","L2","L3","M1","M2","M3","M4","M5"]
            
            #self.update({symbol+'-'+line : 0.0 for line in lines})
            
            for fl in element.findall("fluorescence_line"):
                line_type = fl.attrib["type"]
                total_counts = float(fl.attrib["total_counts"])
                
                key = symbol + '-' + line_type[:2]
                key_sum = key[:-1]
                
                if key_sum not in self:
                    self[key_sum] = total_counts
                    self[key] = total_counts
                else:
                    self[key_sum] += total_counts
                    if key not in self:
                        self[key] = total_counts
                    else:
                        self[key] += total_counts

class SyntheticSpectraXRF(Spectra):
    
    def __init__(self, layers_names = None):
        super().__init__()
        self.layers_names = layers_names
    
    def from_file(self, filepath):
#         try:
#             xml_data = et.parse(filepath)
#         except et.ParseError:
#             print(f"Error while parsing file:\n{filepath}")
        
        xml_data = et.parse(filepath)
        
        interaction_number = xml_data.find('./spectrum_conv/channel').findall('counts').__len__()

        self.nchannels = int(xml_data.find("./xmimsim-input/detector/nchannels").text)
        
        #self.time = float(xml_data.find('./xmimsim-input/detector/live_time').text)

        self.counts = empty((self.nchannels))
        self.unconv_counts = empty_like(self.counts)
        self.energy = empty_like(self.counts)

        convoluted = xml_data.find("spectrum_conv")
        unconvoluted = xml_data.find("spectrum_unconv")

        for i, d in enumerate(zip(convoluted.findall(".//energy"),
                                  convoluted.findall(f".//counts[@interaction_number = '{interaction_number}']"),
                                  unconvoluted.findall(f".//counts[@interaction_number = '{interaction_number}']"))):

            self.energy[i] = float(d[0].text)
            self.counts[i] = float(d[1].text)
            self.unconv_counts[i] = float(d[2].text)
        
        self.layers = Layers(xml_data, self.layers_names)    
        self.labels = Labels(xml_data)
        
        if not self.layers_names:
            self.layers_names = list(self.layers.keys())
        
        return self
        
    
    @property
    def reference_layer(self):
        return self.layers.reference_layer
    
    @property
    def reflayer_name(self):
        return self.layers.reflayer_name
    
    def rescale(self, scale_factor):
        
        self.counts *= scale_factor
        self.unconv_counts *= scale_factor
        self.time *= scale_factor
        
        for k,v in self.labels.items():
            self.labels[k] = v*scale_factor
        
        return self
    
    def expand_elements(self, elements, layer = None):
        
        def _expand(_layer, elements):
            res = []
            for e in elements:
                if e in _layer:
                    res += [e]
                else:
                    res += ["na"]
            return res
        
#         elements = set()
#         for layer in self.layers:
#             elements.update(layer.elements)
        
        if layer:
            out = _expand(self.layers[layer], elements)
        else:
            out = [_expand(self.layers[layer], elements) for layer in self.layers]
        
        return out

