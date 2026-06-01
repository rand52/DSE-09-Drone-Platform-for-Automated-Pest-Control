import meshio
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components

# ----- Fusion 360 example result export command -----
# SimResults.ExportActiveResults C:\Users\Lenovo\Desktop\Sim_results\duct_headon_velocity.vtu

# ---------------- Read file ----------------
file = "duct_headon_velocity.vtu"
path = r"C:\Users\Lenovo\Desktop\Sim_results"

m = meshio.read(path + "\\" + file)

# Result field (grabs whatever was exported)
field_name = list(m.point_data.keys())[0]

# Get mesh parameters
points = m.points
elements = m.cells[0].data   # tetra10 mesh
# Each element is comprised of points:  element = [points 2, 5, 7]

# Get the results tensor for each point
result = m.point_data[field_name]

# ---------------- Find disconnected bodies ----------------
n_nodes = len(points)
# Boolean connectivity matrix for the mesh points
# adj[i, j] = 1 means point i is connected to point j.
adj = lil_matrix((n_nodes, n_nodes), dtype=np.uint8)

# Fill in the connections between points inside each element
for elem in elements:
    adj[np.ix_(elem, elem)] = 1

# Scipy graph magic to identify connected graphs, where the
# mesh points are the graph nodes. In this case disconnected
# graphs are separate bodies.
n_bodies, labels = connected_components(adj, directed=False)

print(f"Found {n_bodies} bodies")

# ---------------- Identify wall ----------------
body_mean_velocity = []

for i in range(n_bodies):
    ### Nodes belonging to current body ###
    body_nodes = np.where(labels == i)[0]

    ### Elements belonging fully to this body ###
    # Mask [n_elements, nodes_per_element]. If a node/point
    # is part of the body the entry is true
    mask = np.isin(elements, body_nodes)
    # If all entries on the row are true the element
    # belongs to the body (all points of the element are
    # points of the body)
    mask = np.all(mask, axis=1)
    body_elements = elements[mask]

    ### Mean element velocity ###
    # Get the mean for each element (mean of the
    # comprising points)
    v = result[body_elements].mean(axis=1)
    # Get the mean for the body (mean if the
    # comprising elements)
    v = v.mean()

    body_mean_velocity.append(v)

    print(f"Body {i}: mean velocity = {v:.6f}")


# Wall = body with the smallest absolute velocity
wall_body = np.argmin(np.abs(body_mean_velocity))

print(f"\nDetected wall: Body {wall_body}")

# ---- Keep non-wall bodies for average velocity ----
# Get nodes belonging to the wall body
wall_nodes = np.where(labels == wall_body)[0]
# Basically do the same as above when finding the
# elements belonging to the body. Then ~ flips
# True to False and vice versa, so now you get
# the elemensts that don't belong to the wall body
keep_mask = ~np.all(np.isin(elements, wall_nodes), axis=1)
drone_elements = elements[keep_mask]

# ---------------- Compute velocities ----------------
# Get the mean for each element (mean of the
# comprising points)
element_velocity = result[drone_elements].mean(axis=1)

print("\nNon-wall bodies only:")
print(f"Min mesh element velocity: {element_velocity.min():.6f}")
print(f"Max mesh element velocity: {element_velocity.max():.6f}")
print(f"Average mesh element velocity: {element_velocity.mean():.6f}")