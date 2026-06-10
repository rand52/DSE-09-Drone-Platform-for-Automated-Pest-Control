import meshio
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components

# ----- Fusion 360 example result export command -----
# SimResults.ExportActiveResults C:\Users\Lenovo\Desktop\Sim_results\duct_headon_velocity.vtu

# ---------------- Read file ----------------
file = "whoop_bottom_plastic.vtu"
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
# Filter out orphaned nodes (disconnected single nodes not
# part of any  that can for some reason be generated)
used_nodes = np.unique(elements.flatten())
# Create a mapping from old node indices to new ones
node_mapping = {}
for new_idx, old_idx in enumerate(used_nodes):
    node_mapping[old_idx] = new_idx
# Remap all element node indices to use the new numbering
elements_remapped = []
for elem in elements:
    new_elem = []
    for node in elem:
        new_elem.append(node_mapping[node])
    elements_remapped.append(new_elem)

# Get the fields from the actually used points
points = points[used_nodes]
elements = np.array(elements_remapped)
result = result[used_nodes]
n_nodes = len(used_nodes)

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

# ---------------- Identify frame ----------------
body_n_yielded_elements = []
n_body_elements = []
body_mean_plastic_strain = []


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

    ### Filter out orphaned nodes bodies ###
    # orphan nodes = disconnected single nodes not
    # part of any  that can for some reason be generated
    # Hence disregard bodies comprised of no elements
    # but just a node.
    # Skip fake bodies with no elements
    if len(body_elements) == 0:
        continue

    ### Mean element strain ###
    # Get the mean  strain for each element in the body
    # (mean of the comprising points)
    eps = result[body_elements].mean(axis=1)

    # For each element, if the average of the strain across its
    # nodes is > 0 it has yielded
    # This is an array of booleans (0 or 1) if it has yielded or not
    yield_mask = eps > 0

    # Get the number of yielded elements
    n_yielded = np.sum(yield_mask)

    # Compute the mean strain of the yielded elements
    if n_yielded > 0:
        mean_yielded_strain = eps[yield_mask].mean()
    else:
        mean_yielded_strain = 0.0

    # Save
    body_n_yielded_elements.append(n_yielded)
    body_mean_plastic_strain.append(mean_yielded_strain)
    # total body elements
    n_body_elements.append(len(body_elements))

    print(f"Body {i}: Num yielded elements = {n_yielded}/{len(body_elements)}, "
          f"mean plastic strain of yielded elements = {mean_yielded_strain:.6f}")

# Go to np array
body_n_yielded_elements = np.array(body_n_yielded_elements)

# Find the frame as the body that has yielded
# elements (potentially)
if np.max(body_n_yielded_elements) == 0:
    print("No yielding detected in any body.")
    exit()
yielding_body = np.argmax(body_n_yielded_elements)
print(f"\nDetected yielded frame: Body {yielding_body}")


# ---------------- Print results ----------------
print("\nYielding body (Frame) statistics:")
print(f"Yielded elements: {body_n_yielded_elements[yielding_body]}/"
      f"{n_body_elements[yielding_body]}"
      f"({body_n_yielded_elements[yielding_body]/n_body_elements[yielding_body]*100:.3f} %)")
print(f"Mean plastic strain of yielded elements: {body_mean_plastic_strain[yielding_body]:.6f}")