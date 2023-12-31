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


def calc_group_distance(group):
  """
  
  Calculate the hamming distance between all arrays in the group.

  Params
  ------
  group: array of binary strings to compare.

  Returns
  -------
  distance: float - Distance between all strings.
  """

  distance = 0
  for i, first in enumerate(group):
    for j, second in enumerate(group):
      if i < j:
        distance += hamming_distance(first, second)

  return distance


def find_optimal_group(arrays):
  """
  
  Given all arrays, find the group of strings that are the least similar.
  Uses hamming distance to calculate string similarity.

  Params
  ------
  arrays: array of binary strings

  Returns
  -------
  group: array - The 3 least similar bit strings of all combinations.
  """

  distances = []
  
  for i, first in enumerate(arrays):
    for j, second in enumerate(arrays):
      for k, third in enumerate(arrays):
        if i < j and j < k:
          distance = calc_group_distance([first, second, third])
          distances.append((i, j, k, distance))

  sorted_distances = sorted(
    distances, key=lambda tup: tup[-1], reverse=True
  )

  indexes = sorted_distances[0][:-1]
  print("Optimal group of memories: {}".format(indexes))

  group = []
  for index in indexes:
    group.append(arrays[index])

  group = np.array(group)
  
  return group


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