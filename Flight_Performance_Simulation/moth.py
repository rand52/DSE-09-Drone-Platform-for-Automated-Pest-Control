"""
This code basically loads in moth data we obtained from PATS
In their data, the origin is the camera, so positions are relative from there
Camera is about 3m in the air, we will take this into account
"""

from __future__ import annotations
import numpy as np
import pandas as pd

class MothTrajectory:
    """
    From the csv, we get a moth trajectory
    CHANGELOG BELOW:
    CHANGE 1: First iteration for the code made
    CHANGE 2: Pandas used for faster file loading
    """
    def __init__(self, path: str, require_valid: bool = True, smoothed: bool = True ) -> None:
        # Check whether smoothed is used 
        # Typically I think we would want smoothed but we keep the option for group members
        if smoothed:
            px, py, pz = ("sposX_insect", "sposY_insect", "sposZ_insect")
        else:
            px, py, pz = ("posX_insect", "posY_insect", "posZ_insect")
        
        columns_we_need = ["elapsed", "pos_valid_insect", px, py, pz, "svelX_insect", "svelY_insect", "svelZ_insect"]
        
        #Loading the dataframe, but only the columns we need
        df=pd.read_csv(path, sep=";", on_bad_lines="skip", index_col=False, usecols=columns_we_need)

        #filter out insect positions that are invalid
        #e.g. when camera loses track
        if require_valid:
            df=df[df["pos_valid_insect"]==1]
        
        #to perform mathematical operations (for velocity) we need at least two samples 
        #of course we want more but we want to stop here otherwise the code might crash
        if len(df)<2:
            raise ValueError(f"Not enough valid moth positions in {path}")
        #sort chronologically
        df=df.sort_values(by="elapsed")
        
        #Extract our data to numpy arrays so we can import in mujoco
        self.t = (df["elapsed"] - df["elapsed"].iloc[0]).to_numpy()
        self.pos = df[[px, py, pz]].to_numpy()
        self.vel = df[["svelX_insect", "svelY_insect", "svelZ_insect"]].to_numpy()
        self.duration = float(self.t[-1])
        self.start_pos = self.pos[0].copy()
        self.end_pos = self.pos[-1].copy()

        
