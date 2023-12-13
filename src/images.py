import os
import cv2
import numpy as np

def read_images(number=10):
  """
  Read image files and return numpy array of memories for a HNN.

  Params
  ------
  number: int - Number of images to load in, defaults to 10.

  Returns
  -------
  memories: array - Array of images for the network to remember.
  """
  memories = []
  count = 0
  for dirname, _, filenames in os.walk('Digits'):
    for filename in filenames:
      filepath = os.path.join(dirname, filename)
      img = cv2.imread(filepath, 0)
      img_norm = cv2.normalize(img, None, -1, 1.0, cv2.NORM_MINMAX, 
                               dtype=cv2.CV_64F)
      img_flat = np.array(img_norm).flatten()
      memories.append(img_flat)
      count += 1
      if count >= number:
        break

  memories = np.array(memories)

  return memories
