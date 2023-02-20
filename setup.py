#!/usr/bin/env python
from distutils.core import setup

setup(name='XRDXRFutils',
    version='0.21',
    description='XRD and XRF utilities',
    author='Zdenek Preisler',
    author_email='z.preisler@gmail.com',
    packages=['XRDXRFutils'],
    package_dir={'XRDXRFutils' : 'XRDXRFutils'},
    package_data={'XRDXRFutils' : ['mendeleev.dat']}
    )
