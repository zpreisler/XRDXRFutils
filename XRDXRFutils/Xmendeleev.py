import os
from collections import namedtuple

class Xmendeleev():
    """Provide elements symbol atomic number and density"""
    def __init__(self, datafile = None):
        if not datafile:
            if os.name == 'posix':
                self.datafile = '/home/rosario/xmimsim/xsimspe/mendeleev.dat'
            else:
                self.datafile = 'C:\\User\\XRAYLab\\rosario_sim\\xsimspe\\mendeleev.dat'
        else:
            self.datafile = datafile
        if not os.path.exists(self.datafile):
            raise FileNotFoundError('mendeleev.dat not found. Data file required')
        self.data = {}
        self.mendeleev = namedtuple("mendeleev", ["symbol", "atomic_number", "density"])
        with open(self.datafile) as fin:
            for line in fin:
                S, A, D = line.split()
                self.data[S] = self.mendeleev(symbol = S, atomic_number = int(A), density = float(D))
    
    def get_element(self, elem):
        if isinstance(elem, str):
            try:
                elem = int(elem)
                return list(self.data.values())[elem - 1]
            except:
                if elem in self.data.keys():
                    return self.data[elem]
                else:
                    raise ValueError(f'{elem} element not found')
        elif isinstance(elem, int):
            return list(self.data.values())[elem - 1]
            
