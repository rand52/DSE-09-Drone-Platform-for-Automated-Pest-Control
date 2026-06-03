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
        #ASSUME FOR NOW THAT COLUMN 1 IS ANGLES AND COLUMN 2 is CD
        self.angles_ref = df.iloc[:,0].to_numpy()
        self.cds_ref = df.iloc[:,1].to_numpy()

        #get the body from mujoco, apparently it is written in C and has an integer string id 
        #Bit cumbersome to fetch it but here we are

        self.body_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        #also, when we misspell or it does not exist, the simulation crashes, to avoid having to set up the mujoco viewer every time
        # we raise the error in python, if it does not exist, value that is assigned is -1, so this is the way to check
        if self.body_id<0:
            raise ValueError(f"Body '{body_name}' not found in model.")

        #necessary to avoid enormous runtimes
        self._vel6_world = np.zeros(6, dtype=float)
        self._vel6_local = np.zeros(6, dtype=float)

        def body_velocity(self) -> tuple[np.ndarray, np.ndarray]:
            "We fetch both the world and local velocities"
            "The world one can be used for target tracking and stuff"
            "The local one for control aerodynamics and whatever"
            #the zero at the end fetches velocity in world axes
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.body_id,self._vel6_world,0)
            #same goes for the 1 at this end, but then local
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.body_id, self._vel6_local,1)
            return self._vel6_world[3:6], self._vel_local[3:6] 
        
        def compute_drag(self) -> np.ndarray:
            "returns drag in the world frame, which is necessary for the simulation to work properly"
            #we get the world and local velocities
            v_world, v_local = self.body_velocity()
            speed = float(np.linalg.norm(v_world))
            
            #for low speeds we ignore
            if speed < 1e-6:
                return np.zeros(3)
            #Here we get the angle, x points in thrust direction, cos vx/v gives theta
            v_x_local=v_local[0]
            angle_rad = np.arccos(np.clip(v_x_local/speed, -1.0, 1.0))
            angle_deg=np.degrees(angle_rad)

            cd_current=np.interp(angle_deg, self.angles_ref, self.cds_ref)

            #drag 
            return -0.5*self.rho*self.area*cd_current*speed*v_world
        
        
        




