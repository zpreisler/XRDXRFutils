from .database import Phase, PhaseList, DatabaseXRD
from .data import DataXRF, DataXRD
from .spectra import SpectraXRF, SpectraXRD, FastSpectraXRD
from .calibration import Calibration
from .gaussnewton import GaussNewton
from .gammasearch import GammaSearch, GammaMap
from .gammasearch_secondary import GammaSearch_Secondary, GammaMap_Secondary
from .chisearch import ChiSearch, ChiMap
from .utils import snip, convolve, convolve3d

from os.path import isdir, exists
from os import makedirs, remove
from shutil import rmtree
from pathlib import Path
from multiprocessing import Pool

import re
import h5py
from glob import glob
from math import ceil

from numpy import (linspace, concatenate, append, sqrt, log, sin, cos, pi, deg2rad, histogram, array,
    unravel_index, savetxt, nan, isnan, flip, sum, average, amax, nanmax, nanmin, nanmean, nanargmax, maximum,
    minimum, arange, empty, newaxis, stack, clip, quantile, nanquantile, ones, zeros, absolute, rot90,
    asarray, loadtxt, where, argwhere)

from pandas import DataFrame, read_csv, concat

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks

from matplotlib.pyplot import (show, close, sca, fill_between, legend, imshow, subplots, plot,
    xlim, ylim, xlabel, ylabel, cm, title, scatter, colorbar, figure, vlines, savefig, get_cmap, hist)
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from matplotlib.markers import MarkerStyle
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap

from PIL import Image



def f_linear(x, a, b):
    return a*x + b


def f_loss(x, t, y):
    return (x[0]*t + x[1]) - y


def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


def read_raw_XRD(path_xrd, filename_scanning = 'Scanning_Parameters.txt', filename_calibration = 'calibration.ini', filename_h5 = 'xrd.h5'):
    return (
        DataXRD()
        .read_params(path_xrd + filename_scanning)
        .read(path_xrd)
        .calibrate_from_file(path_xrd + filename_calibration)
        .remove_background()
        .save_h5(path_xrd + filename_h5)
    )


def correct_point(experimental_phases, idx_phase, gm, x, y):
    gn = gm.get_pixel(x, y)[idx_phase]
    phase = gn.make_phase()
    phase.set_name('created_%d'%idx_phase)
    phase.set_point(gm.get_index(x, y))
    experimental_phases[idx_phase] = phase

    
def rename_phase_in_database(database, name_old, name_new):
    for p in database[name_old]:
        p['_chemical_name_mineral'] = name_new
    database[name_new] = database[name_old]
    del database[name_old]


def find_element(element, labels, allow_loose = False):
    for j, label in enumerate(labels):
        if (element + '_') in label:   # search for the given string + '_' in XRF label
            return j
    if allow_loose:
        for j, label in enumerate(labels):
            if element in label:   # search for the given string in XRF label
                return j
    return None


def settings_lims(ax, x_min, x_max, y_min, y_max):
    if (x_min is not None):
        x_min -= 0.5
    if (x_max is not None):
        x_max += 0.5
    if (y_min is not None):
        y_min -= 0.5
    if (y_max is not None):
        y_max += 0.5
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def settings_plot(im, ax, x_min, x_max, y_min, y_max, position_colorbar, label_colorbar = None, powerlimits = None):
    settings_lims(ax, x_min, x_max, y_min, y_max)
    if (powerlimits is not None):
        formatter = ScalarFormatter(useMathText = True)
        formatter.set_scientific(True)
        formatter.set_powerlimits(powerlimits)
        cb = colorbar(im, ax = ax, cax = ax.inset_axes(position_colorbar), format = formatter)
        cb.ax.yaxis.set_offset_position('left')
    else:
        cb = colorbar(im, ax = ax, cax = ax.inset_axes(position_colorbar))
    if (label_colorbar is not None):
        cb.set_label(label_colorbar)


def phases_from_file(filename, database):
    phases = []
    df = read_csv(filename, header = None)
    
    for i in range(df.shape[0]):
        name = df.iloc[i][0]
        index_sample = df.iloc[i][2]
        if name in database.keys():
            phase = database[name][index_sample]
            phase.set_label(f'{name} {index_sample}')
            phases.append(phase)
        else:
            print(f'Could not find phase \'{name}\' in the database.')

    print('Loaded phases: ' + ', '.join([p.label for p in phases]))
    return phases


def is_element_in_formula(element, formula):
    if element == '':
        return False
    else:
        return re.search(element + '[\d\s]|' + element + '$', formula)


def clean_phase_name(name):
    name_clean = name
    for c in ['/', '\\']:
        name_clean = name_clean.replace(c, '_')
    return name_clean

def shorten_string(s, length_max):
    if len(s) > length_max:
        s_short = s[:(length_max - 3)] + '...'
    else:
        s_short = s
    return s_short


rcParams.update({
    'image.origin': 'lower'
})