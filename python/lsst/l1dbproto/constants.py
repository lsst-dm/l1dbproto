"""
Small bunch of constants used in simulation.
"""

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import math

#-----------------------------
# Imports for other modules --
#-----------------------------

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

FOV_deg = 3.5
FOV_rad = FOV_deg * math.pi / 180

N_TRANS_PER_VISIT = 100    # average number of transients per visit
N_FALSE_PER_VISIT = 5050   # average number of false positives per visit

HTM_LEVEL = 20
HTM_MAX_RANGES = 40
