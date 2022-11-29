# setup.py, needed to be able to pip-install the package via git+https://github.com/vzakharov/jukwi.git

import os
import pkg_resources
from setuptools import setup, find_packages

setup(
  name='jukwi', # Name of the package
  py_modules=['server', 'download'], # Name of the python files you want to be able to import
  version='0.1', # Version number
  description='Jukebox inference server', # Short description
  author='Vova Zakharov', # Your name
  packages=find_packages(), # Find all packages in the current directory
  install_requires=[
    str(requirement) for requirement in pkg_resources.parse_requirements(open('requirements.txt'))
  ], # List of python packages to be installed with this package
  include_package_data=True, # Whether to include any data files inside your packages
)