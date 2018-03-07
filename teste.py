import numpy as np
import pandas as pd
from lib.algaas import BarriersWellSandwich

quantum_well = BarriersWellSandwich(b_l=1, d_l=1, w_l=10, 
                                    b_x=0.4, d_x=0.4, w_x=0.0, 
                                    surround=1, offset='well', 
                                    gap_distrib=(0.7, 0.3))

quantum_well.calculate_eigenstates(n=3)
print(np.ptp(quantum_well.v_ev))