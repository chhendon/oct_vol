#!/usr/bin/env python3

import numpy as np
from scipy.spatial import cKDTree

def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

# Function to calculate the volume of an octahedron
def octahedron_volume(p1, p2, p3, p4, p5, p6):
    # Define the vertices of the octahedron
    vertices = [p1, p2, p3, p4, p5, p6]
    volume = 0
    reference_point = vertices[0]
    furthest_point_index = np.argmax([distance(reference_point, point) for point in vertices])
    furthest_point = vertices[furthest_point_index]
    #Reduce vertices to equatorial points
    vertices.pop(furthest_point_index)
    vertices.pop(0)
    eq_reference = vertices[0]
    eq_furthest_index = np.argmax([distance(eq_reference, point) for point in vertices])
    eq_furthest = vertices[eq_furthest_index]
    #Reduce vertices to opposite points
    vertices.pop(eq_furthest_index)
    vertices.pop(0)
    v0 = np.array(reference_point)
    v3 = np.array(furthest_point)
    v1 = np.array(eq_reference)
    v2 = np.array(vertices[0])
    tetrahedron_volume = abs((1 / 6) * np.linalg.det([v1 - v0, v2 - v0, v3 - v0]))
    volume += tetrahedron_volume
    v2 = np.array(vertices[1])
    tetrahedron_volume = abs((1 / 6) * np.linalg.det([v1 - v0, v2 - v0, v3 - v0]))
    volume += tetrahedron_volume
    v1 = np.array(eq_furthest)
    tetrahedron_volume = abs((1 / 6) * np.linalg.det([v1 - v0, v2 - v0, v3 - v0]))
    volume += tetrahedron_volume
    v2 = np.array(vertices[0])
    tetrahedron_volume = abs((1 / 6) * np.linalg.det([v1 - v0, v2 - v0, v3 - v0]))
    volume += tetrahedron_volume

    return volume


# Read lines from file
with open('structure_out.xyz', 'r') as file:
    lines = file.readlines()[2:]

# Extract strings and coordinates
strings = [line.strip() for line in lines]
coordinates = np.array([list(map(float, line.split()[1:])) for line in lines])
atoms = [list(line.split()[0]) for line in lines]

# Build KD-tree for efficient nearest neighbor search
kdtree = cKDTree(coordinates)
file = open('Ti_octahedron_volumes.txt', 'w')
ti_all_vol = []

# Iterate through each point and find its 6 nearest neighbors
for i, point in enumerate(coordinates):
    # Query the KD-tree to get the 7 nearest neighbors (including the point itself)
    _, neighbors_indices = kdtree.query(point, k=7)

    # Extract the coordinates of the neighbors
    neighbors = coordinates[neighbors_indices[1:]]

    # Extract the string associated with the current point
    current_string = atoms[i]

    # Calculate the volume of the octahedron formed by the point and its 6 neighbors
    octahedron_vol = octahedron_volume(*neighbors)
    if current_string == ['T','i']:
        ti_all_vol.append(octahedron_vol)
        print(f"String: {current_string}, Point: {point}, Octahedron Volume: {octahedron_vol}")
        file.write(str(octahedron_vol) + '   ' + str(neighbors_indices) + '\n')

average = np.mean(ti_all_vol)
std_dev = np.std(ti_all_vol)
file.write('\nAverage: ' + str(average) + '\nStandard deviation: ' + str(std_dev))
file.close()
