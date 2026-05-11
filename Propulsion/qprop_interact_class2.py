import subprocess
import os
import numpy as np

class QMILInterface:
    def __init__(self, name="Propeller"):
        self.name = name
        self.n_blades = 3
        # Airfoil characteristics
        self.cl_params = {"cl0": 0.4171, "cl_a": 5.19, "cl_min": -0.35, "cl_max": 1.4}
        self.cd_params = {"cd0": 0.15, "cd2u": 0.018, "cd2l": 0.06, "clcd0": 0.49}
        self.re_params = {"re_ref": 50000, "re_exp": -0.27}
        # Design distribution (r/R and CL)
        self.dist_r_R = [0.0, 0.5, 1.0]
        self.dist_cl = [0.0, 0.5, 0.4]
        # Operating points
        self.radii = {"hub": 0.00075, "tip": 0.020}
        self.op_point = {"vel": 0, "rpm": 40000.0}
        self.targets = {"thrust": 0.4, "power": 0} # Use 0 for the one not specified
        # Design options
        self.ldes = 0  # 0=Min Induced Loss, 2=Windmill Max Power
        self.kqdes = 0
        self.n_out = 30

    def write_input_file(self, filename="qmil.inp"):
        with open(filename, 'w') as f:
            f.write(f"{self.name}\n\n")
            f.write(f" {self.n_blades} ! Nblades\n\n")
            f.write(f" {self.cl_params['cl0']} {self.cl_params['cl_a']} ! CL0 CLa\n")
            f.write(f" {self.cl_params['cl_min']} {self.cl_params['cl_max']} ! CLmin CLmax\n\n")
            f.write(f" {self.cd_params['cd0']} {self.cd_params['cd2u']} {self.cd_params['cd2l']} {self.cd_params['clcd0']} ! CD0 CD2u CD2l CLCD0\n")
            f.write(f" {self.re_params['re_ref']} {self.re_params['re_exp']} ! REref REexp\n\n")
            f.write(f" {' '.join(map(str, self.dist_r_R))} ! XIdes\n")
            f.write(f" {' '.join(map(str, self.dist_cl))} ! CLdes\n\n")
            f.write(f" {self.radii['hub']} ! hub radius\n")
            f.write(f" {self.radii['tip']} ! tip radius\n")
            f.write(f" {self.op_point['vel']} ! speed\n")
            f.write(f" {self.op_point['rpm']} ! rpm\n\n")
            f.write(f" {self.targets['thrust']} ! Thrust\n")
            f.write(f" {self.targets['power']} ! Power\n\n")
            f.write(f" {self.ldes} {self.kqdes} ! Ldes KQdes\n\n")
            f.write(f" {self.n_out} ! Nout\n")
        return filename

class QPROPInterface:
    def __init__(self, motor_name="Motor"):
        self.motor_name = motor_name
        self.motor_type = 1 # Type 1 = Standard brushed DC
        # Motor Params (R, Io, Kv)
        self.params = [0.221, 0.38, 1700]

    def write_motor_file(self, filename="motor.dat"):
        with open(filename, 'w') as f:
            f.write(f"{self.motor_name}\n\n")
            f.write(f" {self.motor_type} ! Motor type\n\n")
            for p in self.params:
                f.write(f" {p}\n")
        return filename

def run_software(qmil_obj, qprop_obj, vel, rpm, volt, output_file=os.path.join(os.getcwd(), "results", "qprop_output.txt")):
    """Executes the QMIL -> QPROP workflow"""
    
    # 1. Generate QMIL files and run
    qmil_in = qmil_obj.write_input_file("design.inp")
    prop_file = "generated.prop"
    
    print(f"--- Running QMIL ---")
    subprocess.run(["qmil", qmil_in, prop_file])
    
    # 2. Generate Motor file
    motor_file = qprop_obj.write_motor_file("motor.dat")
    
    # 3. Run QPROP (Single point example)
    print(f"--- Running QPROP Analysis ---")
    # Command: qprop propfile motorfile Vel Rpm Volt
    result = subprocess.run(
        ["qprop", prop_file, motor_file, str(vel), str(rpm), str(volt)],
        capture_output=True, text=True
    )
    with open(output_file, 'w') as f:
        f.write(result.stdout)
    print(f"Results saved to {output_file}")
    
    return result.stdout

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # Define variables in code
    design = QMILInterface("MyProp_v1")
    
    motor = QPROPInterface("Speed-400")
    motor.params = [0.078, 2.3, 22000] # Resistance, Io, Kv
    
    # Run the simulation
    output = run_software(design, motor, vel="0", rpm="20000,40000/8", volt=0)
    
    print("\nQPROP Output Results:")
    print(output)