"""The contents of this package allow users to reproduce our microtubule catastrophe analysis. It includes:
Our methods for parsing the raw data
Methods for performing exploratory analysis
Statistical analysis modules (bootstrapping, hypothesis testing, MLE analysis, and model assessment)
"""
from .parsing import *
from .exploratory_analysis import *
from .bootstrapping import *
from .MLE_analysis import *
from .model_assessment import *
__author__ = 'Joeyta Banerjee'
__email__ = 'jbanerje@caltech.edu'
__version__ = '0.0.1'
