"""
In this drag model we will simulate the drag characteristics, 
MuJoCo does not simulate drag naturally, so we have to give it as input
Cd values are straight from team members

Quadratic Drag model is used:
D=1/2 rho V^2 Sref Cd


NOTE: Until 60 degrees, the actual CAD model of the drone is used, 
after that, a cylinder is used (but the Cd values are still from CFD)
NOTE: No wind is assumed
NOTE: Sref=0.004, thats what the groupmembers normalised with, and is what we will use as well
NOTE: Tether drag is neglected!
"""

from __future__ import annotations
import os
import mujoco
import numpy as np
import pandas as pd

class AeroEngine:
    """
    Calculates drag forces on the drone as described above using the quadratic model
    NOTE: rho and Sref are in the top line of init, change if needed.
    CHANGELOG
    CHANGE 1: Original code made
    CHANGE 2: Removed wind
    CHANGE 3: Import csv file for Cd and read it using pandas
    CHANGE 4: Overall integration with other team members
    """
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, cd_csv_path: str, body_name: str="drone", rho: float=1.225, area: float=0.004) -> None:
        self.model= model
        self.data = data
        self.rho = float(rho)
        self.area = float(area)

        #load the CFD data, which we will put in a csv file
        if not os.path.exists(cd_csv_path):
            raise FileNotFoundError(f"File with Cd's not found at {cd_csv_path}")
        
        df=pd.read_csv(cd_csv_path)
        #ASSUME FOR NOW THAT COLUMN 1 IS ANGLES AND COLUMN 2 is CD @Maxschuurkes

