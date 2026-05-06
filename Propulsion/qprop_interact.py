import subprocess
import os

class QMILInterface:
    def __init__(self, name="Propeller"):
        self.name = name
        self.n_blades = 3
        # Airfoil characteristics
        self.cl_params = {"cl0": 0.3931, "cl_a": 9.5157, "cl_min": -0.3177, "cl_max": 1.3863}
        self.cd_params = {"cd0": 0.01593, "cd2u": 0.02000, "cd2l": 0.02882, "clcd0": 1.2127}
        self.re_params = {"re_ref": 100000, "re_exp": -0.5}
        # Design distribution (r/R and CL)
        self.dist_r_R = [0.0, 0.5, 1.0]
        self.dist_cl = [0.0, 0.5, 0.4]
        # Operating points
        self.radii = {"hub": 0.0044, "tip": 0.044}
        self.op_point = {"vel": 10, "rpm": 30000.0}
        self.targets = {"thrust": 2.4525, "power": 0} # Use 0 for the one not specified
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

def run_software(qmil_obj, qprop_obj, vel, rpm, volt):
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
    
    return result.stdout

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # Define variables in code
    design = QMILInterface("MyProp_v1")
    # design.n_blades = 3
    # design.op_point['rpm'] = 300.0
    # design.targets['power'] = 450.0  # Design for 450W
    
    motor = QPROPInterface("Speed-400")
    motor.params = [0.221, 0.38, 1700] # Resistance, Io, Kv
    
    # Run the simulation
    # Analyzes performance at 10 m/s, using the motor at 12V
    output = run_software(design, motor, vel=0.0, rpm=0, volt=23.5)
    
    print("\nQPROP Output Results:")
    print(output)