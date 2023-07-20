# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

setup(
    name='virtual_eve',
    version='0.1.0',
    description='Virtual EVE irradiance prediction model',
    long_description=readme,
    author='FDL-X 2023 ARD EUV Team',
    author_email='me@kennethreitz.com',
    url='https://github.com/kennethreitz/samplemod',
    packages=find_packages(exclude=('old', 'data', 'figures', 'models', 'notebooks', 'scripts', 'tests'))
)