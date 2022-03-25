from scipy.optimize import curve_fit
from numpy import pi, arctan
from numpy import loadtxt, frombuffer, array, asarray, linspace, arange, trapz, flip, stack, where
from scipy.interpolate import interp1d
from matplotlib.pyplot import plot, xlim, ylim, xlabel, ylabel

from PIL import Image

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

    def from_parameters(self, opt):
        self.opt = array(opt, dtype = 'float')
        self.parent.opt = self.opt.copy()
        return self

    def from_file(self, filename):
        self.metadata['filename'] = filename
        self.x, self.y = loadtxt(filename, unpack = True, dtype = 'float')
        opt, opt_var = curve_fit(self.fce, self.x, self.y)
        return self.from_parameters(opt)

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
