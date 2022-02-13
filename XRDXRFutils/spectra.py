from numpy import loadtxt,arctan,pi,arange,array, asarray
from matplotlib.pyplot import plot
from .utils import snip,convolve
import xml.etree.ElementTree as et

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

class FluorescenceSXRF:
    def __init__(self, symbol, atomic_number, lines ):
        self.symbol = symbol
        self.atomic_number = atomic_number
        self.lines = lines

class SpectraSXRF(Spectra):
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def get_metadata(xml_data):
        reflayer_index = int(xml_data.find("./xmimsim-input/composition/reference_layer").text) - 1
        reflayer = xml_data.findall("./xmimsim-input/composition/layer")[reflayer_index]
        sublayer = xml_data.findall("./xmimsim-input/composition/layer")[reflayer_index + 1]
        
        elements = []
        weight_fractions = []
        for element in reflayer.findall("element"):
            elements.append(int(element.find("atomic_number").text))
            weight_fractions.append(float(element.find("weight_fraction").text))
            
        reflayer_thickness = float(reflayer.find("thickness").text)
        sublayer_thickness = float(sublayer.find("thickness").text)
        
        return elements, weight_fractions, reflayer_thickness, sublayer_thickness
    
    @staticmethod
    def get_fluorescence_lines(xml_data):
        """Generator"""
        flc = xml_data.findall(".//fluorescence_line_counts")
        for element in flc:
            lines = {"K" : 0, "L" : 0, "others" : 0}
            for fl in element.findall("fluorescence_line"):
                line_type = fl.attrib["type"]
                if line_type.startswith("K"):
                    lines["K"] += float(fl.attrib["total_counts"])
                elif line_type.startswith("L"):
                    lines["L"] += float(fl.attrib["total_counts"])
                else:
                    lines["others"] += float(fl.attrib["total_counts"])
                    
            yield FluorescenceSXRF(
                symbol = element.attrib["symbol"],
                atomic_number = element.attrib["atomic_number"],
                lines = lines
            )

    
    def from_file(self, xmso_filename, interaction_number = 2, shape = None):
        xml_data = et.parse(xmso_filename)
        convoluted = xml_data.find("spectrum_conv")
        self.energy = asarray([e.text for e in convoluted.findall(".//energy")], dtype=float)
        self.counts = asarray(
            [c.text for c in convoluted.findall(f".//counts[@interaction_number = '{interaction_number}']")],
            dtype=float,
        )
        if shape:
            self.counts = self.counts.reshape(*shape)
            
        self.channel = arange(self.counts.__len__(),dtype='int16')
        
        self.reflayer_atomic_num, self.weight_fractions, self.reflayer_thickness, self.sublayer_thickness = self.get_metadata(xml_data)
        
        self.fluorescence_lines = list(self.get_fluorescence_lines(xml_data))
        
        return self

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
