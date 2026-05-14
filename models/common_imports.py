import importlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output
from joblib import Parallel, delayed
from matplotlib import cm, colors
from numpy import pi
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from Modeling.models.beam_properties import PiezoBeamParams
from Modeling.models.plotting import animate_field_1d
from Modeling.models import FE_helpers
import Modeling.models.FE3 as FE_module
importlib.reload(FE_helpers)
importlib.reload(FE_module)
FE = FE_module
