import os
import cv2
import numpy as np

def read_images(number=10, size=50):
  """
  Read image files and return numpy array of memories for a HNN.

  Params
  ------
  number: int - Number of images to load in, defaults to 10.
  size: int - Size of images to load, either 28x28, 50x50 or 280x280.

  Returns
  -------
  memories: array - Numpy array of images for the network to remember.
  """
  if size != 28 and size!= 50 and size != 280:
    print("Size: {} not valid. Please enter 28 or 280.".format(size))
    return

  memories = []
  count = 0
  for dirname, _, filenames in os.walk('Digits/{}x{}'.format(size, size)):
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

def read_single_digit(digit, size=50):
  """
  Read and return a single specified digit.

  Param
  -----
  digit: int - Number to read and return
  size: int - Size of images to load, either 28x28, 50x50, or 280x280.

  Returns
  -------
  memory: array - Numpy array containing the digit, shape (1, 784)
  """
  if size != 28 and size!= 50 and size != 280:
    print("Size: {} not valid. Please enter 28 or 280.".format(size))
    return

  filepath = "Digits/{}x{}/{}.png".format(size, size, str(digit))
  img = cv2.imread(filepath, 0)
  img_norm = cv2.normalize(img, None, -1, 1.0, cv2.NORM_MINMAX, 
                           dtype=cv2.CV_64F)
  img_flat = np.array(img_norm).flatten()

  return np.array([img_flat])

def add_rng_noise(img, n):
  """
  Randomly add noise to a given image.

  A cell is picked at random n times and the state of that cell is flipped.

  Params
  ------
  img: np array - A flattened image.

  n: int - Number of times to randomly pick a cell and flip it's state.

  Returns
  -------
  img: np array - The original image with added noise.
  """

  img = np.array(img)
  print("add_rng_noise - img shape: {}".format(img.shape))

  for i in range(n):
    rnd_index = np.random.randint(0, img.shape[0])
    img[rnd_index] = img[rnd_index] * -1

  return img


