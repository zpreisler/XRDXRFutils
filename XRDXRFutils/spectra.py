from numpy import loadtxt, arctan, pi, arange, array, asarray, linspace, zeros, maximum, nanmax, where
from math import ceil
from matplotlib.pyplot import plot
import xml.etree.ElementTree as et
from scipy.interpolate import interp1d
from re import sub as re_sub
import warnings

from .utils import snip, convolve
from .calibration import Calibration


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


    @property
    def length(self):
        return len(self.counts)


    @staticmethod
    def fce_calibration(x, a, b, c):
        """
        Calibration function 
            x is a channel
        """
        return a * x**2 + b * x + c


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


class SyntheticSpectraXRF(Spectra):
    def __init__(self, rl_atnum_list, skip_element = False):
        super().__init__()
        self.nbins = None
        if not isinstance(rl_atnum_list, list):
            raise TypeError('list instance expected for elements list')
        for i,item in enumerate(rl_atnum_list):
            if not isinstance(item, int):
                raise TypeError(f'{item} at index {i} is not integer.\nIntegers are expected for Atomic Numbers')
        self.rl_atnum_list = sorted(rl_atnum_list)
        self.skip_element = skip_element
    
    def set_nbins(self, nbins):
        self.nbins = nbins
    
    @staticmethod
    def rebin(x,y):
        xx = x[::2]
        yp = y[:-1] + y[1:]
        yy = yp[::2]
        return xx, yy
    
    @staticmethod
    def get_metadata(xml_data, rl_atnum_list, skip = False):
        _time = float(xml_data.find('./xmimsim-input/detector/live_time').text)
        reflayer_index = int(xml_data.find("./xmimsim-input/composition/reference_layer").text) - 1
        layers = xml_data.findall("./xmimsim-input/composition/layer")
        reflayer = layers[reflayer_index]
        reflayer_thicknes = float(reflayer.find("thickness").text)
        try:
            sublayer = layers[reflayer_index + 1]
        except IndexError:
            sublayer_thicknes = 0.0
        else:
            sublayer_thicknes = float(sublayer.find("thickness").text)
        
        # elements = np.zeros((len(rl_atnum_list))
        weight_fractions = zeros((len(rl_atnum_list)))
        for element in reflayer.findall("element"):
            atnum = int(element.find("atomic_number").text)
            wf = float(element.find("weight_fraction").text)
            try:
                weight_fractions[rl_atnum_list.index(atnum)] = wf
            except ValueError:
                if skip == False:
                    raise ValueError(f'element with atomic number {atnum} not found in elements list\nSet skip_element = True to ignore this error')
        
        return weight_fractions, reflayer_thicknes, sublayer_thicknes, _time
    
    @staticmethod
    def get_fluorescence_lines(xml_data, time_correction = None):

        class Container:
            def __init__(self, symbol, atomic_number, lines ):
                self.symbol = symbol
                self.atomic_number = atomic_number
                self.lines = lines

        """Generator"""
        flc = xml_data.findall(".//fluorescence_line_counts")
        for element in flc:
            lines = {"K" : 0, "L" : 0, "M" : 0, "others" : 0}
            for fl in element.findall("fluorescence_line"):
                line_type = fl.attrib["type"]
                if line_type.startswith("K"):
                    lines["K"] += float(fl.attrib["total_counts"]) * time_correction if time_correction else float(fl.attrib["total_counts"])
                elif line_type.startswith("L"):
                    lines["L"] += float(fl.attrib["total_counts"]) * time_correction if time_correction else float(fl.attrib["total_counts"])
                elif line_type.startswith("M"):
                    lines["M"] += float(fl.attrib["total_counts"]) * time_correction if time_correction else float(fl.attrib["total_counts"])
                else:
                    lines["others"] += float(fl.attrib["total_counts"]) * time_correction if time_correction else float(fl.attrib["total_counts"])
                    
            yield Container(
                symbol = element.attrib["symbol"],
                atomic_number = element.attrib["atomic_number"],
                lines = lines
            )

    
    def from_file(self, xmso_filename, interaction_number = 2, shape = None, time_correction = None):
        try:
            xml_data = et.parse(xmso_filename)
        except et.ParseError:
            print(f"Error while parsing\n{xmso_filename}")
            return None
        convoluted = xml_data.find("spectrum_conv")
        self.energy = asarray([e.text for e in convoluted.findall(".//energy")], dtype=float)
        if time_correction:
            self.counts = time_correction * asarray(
                [c.text for c in convoluted.findall(f".//counts[@interaction_number = '{interaction_number}']")],
                dtype=float,
            )
        else:
            self.counts = asarray(
                [c.text for c in convoluted.findall(f".//counts[@interaction_number = '{interaction_number}']")],
                dtype=float,
            )
        if shape:
            self.counts = self.counts.reshape(*shape)
        
        if self.nbins:
            self.energy, self.counts = self.rebin(self.energy, self.counts)
            #b = self.energy[1] - self.energy[0]
            #self.counts = self.counts / b
            
        self.channel = arange(self.counts.__len__(),dtype='int16')
        self.weight_fractions, self.reflayer_thicknes, self.sublayer_thicknes, self.time = self.get_metadata(xml_data, self.rl_atnum_list, skip = self.skip_element)
        self.fluorescence_lines = list(self.get_fluorescence_lines(xml_data, time_correction = time_correction))
        return self
    
    def time_correction(self, tc):
        self.counts = self.counts * tc
        for l in self.fluorescence_lines:
            for k, v in l.lines.items():
                l.lines[k] = v * tc
        return self


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
    def fce_calibration(x, a, s, beta):
        """
        XRD calibration function 
            x is a channel
        """
        return (arctan((x + a) / s)) * 180 / pi + beta


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


    def plot(self, *args, **kwargs):
        plot(self.theta, self.intensity, *args, **kwargs)


class FastSpectraXRD(SpectraXRD):
    def __init__(self):
        super().__init__(downsample_max = 3)
