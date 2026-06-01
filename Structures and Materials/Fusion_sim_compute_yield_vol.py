import meshio
import numpy as np

# File path example
file = "Xframe_headon_plastic.vtu"
path = "C:\\Users\\Lenovo\\Desktop\\Sim_results\\"

# Read the VTU file
full_path = path + file
m = meshio.read(full_path)

# Extract nodal plastic strain values for this load step (one scalar per node)
# The sheet is number of result export seps you choose +1, just the format
eps = m.point_data["Strain:Plastic Strain:Step51"]
#eps = m.point_data["Strain:Plastic Strain:Step76"]

# Stack all elements into a single (n_elements x nodes_per_element) array
cells = np.vstack([c.data for c in m.cells])

# For each element, average the strain across its nodes and check if > 0
# If so it has yielded
# This is an array of booleans (0 or 1) if it has yielded or not
plastic = eps[cells].mean(axis=1) > 0

# Print count, total, and percentage

print(f"Total yielded elements: {plastic.sum()}/{len(cells)}")
print(f"In percent: {round(plastic.sum() / len(cells) * 100, 3)} %")