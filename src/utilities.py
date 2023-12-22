import numpy as np

def hamming_distance(vec1, vec2):
  """
  
  Return the hamming distance of two vectors / arrays.

  Params
  ------
  vec1: 1D array - First vector to compare
  vec2: 1D array - Second vector to compare

  Returns
  -------
  distance: float - Proportional distance of the two vectors.
  """

  if len(vec1) != len(vec2):
    print("hamming_distance error - The two vectors are of varying lengths.")
    return
  
  n = len(vec1)

  distance = 0
  for i, bit in enumerate(vec1):
    if bit != vec2[i]:
      distance += 1

  return (distance / n)


def calc_hamming_distance(arrays):
  """
  
  Calculate the hamming distance between every array passed.

  Params
  ------
  arrays: array of binary strings to compare
  
  Returns
  -------
  distances: 2D array of Hamming Distances between all strings.
  """

  distances = np.zeros((len(arrays), len(arrays)))
  for i, vec1 in enumerate(arrays):
    for j, vec2 in enumerate(arrays):
      if i < j:
        distances[i][j] = hamming_distance(vec1, vec2)

  return distances

def sort_hamming_distances(distances):
  """
  
  Take a 2D array of Hamming Distances and return a 1D sorted array.

  Params
  ------
  distances: 2D array - Hamming Distances

  Returns
  -------
  sorted_distances: 1D array - Sorted Hamming Distances in descending order
  """

  distance_tuple = []
  for i in range(len(distances)):
    for j in range(len(distances[i])):
      if distances[i][j] > 0:
        distance_tuple.append((i, j, distances[i][j]))

  sorted_distances = sorted(
    distance_tuple, key=lambda tup: tup[2], reverse=True
  )
  return sorted_distances

