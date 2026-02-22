import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import sys
from pathlib import Path
import importlib
# add Modeling/ to Python path
project_root = Path.cwd().parents[2]
sys.path.append(str(project_root))
import Modeling
importlib.reload(Modeling)
from Modeling.models.plotting import animate_field_1d
from Modeling.models.beam_properties import PiezoBeamParams
from Modeling.models import FE_helpers 
importlib.reload(FE_helpers)
import matplotlib.pyplot as plt
# from Modeling.models.ROM import ROM

import Modeling.models.FE3 as FE_module
importlib.reload(FE_module)
FE = FE_module
# import Modeling.models.ROM as ROM_module
# importlib.reload(ROM_module)
# ROM = ROM_module.ROM

# from FE1 import PiezoBeamFE, frf_sweep, solve_newmark
import numpy as np
from numpy import pi	
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from IPython.display import clear_output
from matplotlib import cm, colors
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

K_i = 1800; K_p = 0.0025; K_c = 0; R_c = 1e3
# K_i = 0; K_p = 100000; K_c = 0; R_c = 1e3
# K_i = 0; K_p = 1e-5; K_c = 0; R_c = 1e3
t_end = 0.5
f0 = 1000
f1 = 3000
dt = 1/f1/20
def v_exc(t, A_exc=50, f0=f0, f1=f1, t_end=t_end):
	return A_exc*np.sin(2*np.pi*(f0+ t*(f1-f0)/t_end) *t)
# N = 40
# hp, hs = 0.31e-3, 0.607e-3 		
params_fe = PiezoBeamParams(
                            hp=0.252e-3, hs=0.51e-3,
                            # hp=0.31e-3, hs=0.607e-3,
                            d31= -1.48e-10,eps_r=1700,
							# rho_p=8000,
							# omega_p=2*pi*100, omega_q=2*pi*1000
                            )
params_fe.zeta_p = 0.0151*8
params_fe.zeta_q = 0.0392*10