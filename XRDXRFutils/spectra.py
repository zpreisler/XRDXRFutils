from numpy import loadtxt,arctan,pi,arange,array, asarray, linspace, zeros
from matplotlib.pyplot import plot
from .utils import snip,convolve
import xml.etree.ElementTree as et
from scipy.interpolate import interp1d

from .calibration import Calibration

class Spectra():
    def __init__(self):
        self.calibration = Calibration(self)
    
    @staticmethod
    def fce_calibration(x,a,b):
        """
        Linear calibration function
        """
        return a * x + b
    
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

class FluoContainer:
    def __init__(self, symbol, atomic_number, lines ):
        self.symbol = symbol
        self.atomic_number = atomic_number
        self.lines = lines

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
        reflayer_thickness = float(reflayer.find("thickness").text)
        try:
            sublayer = layers[reflayer_index + 1]
        except IndexError:
            sublayer_thickness = 0.0
        else:
            sublayer_thickness = float(sublayer.find("thickness").text)
        
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
        
        return weight_fractions, reflayer_thickness, sublayer_thickness, _time
    
    @staticmethod
    def get_fluorescence_lines(xml_data, time_correction = None):
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
                    
            yield FluoContainer(
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
        self.weight_fractions, self.reflayer_thickness, self.sublayer_thickness, self.time = self.get_metadata(xml_data, self.rl_atnum_list, skip = self.skip_element)
        self.fluorescence_lines = list(self.get_fluorescence_lines(xml_data, time_correction = time_correction))
        return self
    
    def time_correction(self, tc):
        self.counts = self.counts * tc
        for l in self.fluorescence_lines:
            for k, v in l.lines.items():
                l.lines[k] = v * tc
        return self

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

        return self

    def fromDataf(self, data, i):

        self.calibrate_from_parameters(data.opt)

        self.counts = data.data.reshape(-1,1280)[i]
        self.rescaling = data.rescaling.reshape(-1)[i]
        self.intensity = data.intensity.reshape(-1,1280)[i]

        self.channel = arange(self.counts.__len__(), dtype = 'int')

        return self

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

    def fromDataf(self, data, i):

        self.opt = data.opt.copy()

        self.counts = data.data.reshape(-1,1280)[i]
        self.rescaling = data.rescaling.flatten()[i]
        self.intensity = data.intensity.reshape(-1,1280)[i]

        self.intensity1 = 0.5 * (self.intensity[::2] + self.intensity[1::2])
        self.intensity2 = 0.5 * (self.intensity1[::2] + self.intensity1[1::2])
        self.intensity3 = 0.5 * (self.intensity2[::2] + self.intensity2[1::2])

        self.channel = arange(1280)
        self.channel1 = arange(0.5,1280,2)
        self.channel2 = arange(1.5,1280,4)
        self.channel3 = arange(3.5,1280,8)

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
