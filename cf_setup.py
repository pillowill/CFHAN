# -*- coding: utf-8 -*-
"""
@Time ： 2023/8/26 22:44
@Auth ： He Yu
@File ：cf_setup.py
@IDE ：PyCharm
@Function ：Function of the script
"""
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name='cf_cy',
      ext_modules=cythonize("cf_cython.pyx"),
      include_dirs=[numpy.get_include()]
      )
