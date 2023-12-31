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
  if not check_image_size(size):
    return

  memories = []
  count = 0
  for dirname, _, filenames in os.walk('Digits/{}x{}'.format(size, size)):
    for filename in filenames:
      filepath = os.path.join(dirname, filename)
      
      image = process_image(filepath)

      memories.append(image)
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
  if not check_image_size(size):
    return

  filepath = "Digits/{}x{}/{}.png".format(size, size, str(digit))
  
  image = process_image(filepath)

  return np.array([image])


def read_odd_digits(number=4, size=50):
  """
  Read in only the odd numbers - 1, 3, 5, 7, 9.
  The theory being the odd digits should use a wider range of cells.
  - Avoiding memories of the 0 and 2 digits overlapping.

  Params
  ------
  number: int - Amount of digits to read in: 1 - 5.
  size: int - Size of images to load, either 28x28, 50x50, or 280x280.


  Returns
  -------
  memories: array - Numpy array of images for the network to remember.
  """

  if not check_image_size(size):
    return
  
  if not (1 <= number <= 5):
    print(
      "Number: {} is out of range. Please enter a value between 1 and 5."
      .format(number)
    )
    return
  
  memories = []
  count = 0
  for dirname, _, filenames in os.walk('Digits/{}x{}'.format(size, size)):
    for i, filename in enumerate(filenames):
      if (i % 2) == 1:
        filepath = os.path.join(dirname, filename)

        image = process_image(filepath)

        memories.append(image)

        count += 1
        if count >= number:
          break

  memories = np.array(memories)

  return memories


def read_given_digits(digits, size=50):
  """
  
  Read in only the digits passed. 
  If Digits = [1, 2] only the 1 and 2 digit will be read and returned.

  Params
  ------
  digits: array - Array of digits to read in.
  size: int - Size of images to load, either 28x28, 50x50, or 280x280.

  Returns
  -------
  memories: array - Numpy array of images for the network to remember.
  """

  if not check_image_size(size):
    return
  
  memories = []
  for digit in digits:
    filepath = 'Digits/{}x{}/{}.png'.format(size, size, digit)

    image = process_image(filepath)

    memories.append(image)

  memories = np.array(memories)

  return memories


def process_image(filepath):
  """
  Read, normalise and flatten image from filepath.

  Params
  ------
  filepath: string - location of image to read.

  Returns
  -------
  image: numpy array - 1D flattened image, with unique values of +1 and -1.
  """

  img = cv2.imread(filepath, 0)
  img_norm = cv2.normalize(img, None, -1, 1.0, cv2.NORM_MINMAX, 
                          dtype=cv2.CV_64F)
  img_flat = np.array(img_norm).flatten()

  img_flat[img_flat <= 0] = -1
  img_flat[img_flat > 0] = 1

  if np.unique(img_flat).shape != (2,):
    print("\nprocess_image error: unique(image) shape does NOT match.")
    print("image unique values: {}\n".format(np.unique(img_flat)))

  return img_flat


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

  for i in range(n):
    rnd_index = np.random.randint(0, img.shape[0])
    img[rnd_index] = img[rnd_index] * -1

  return img


def check_image_size(size):
  """
  Check size of images is valid.

  Params
  ------
  size: int - The size of desired images.

  Returns
  -------
  boolean - True if image size is valid, false if invalid.
  """

  if size != 28 and size!= 50 and size != 280:
    print("\ncheck_image_size - Image size: {} is invalid.".format(size) +
          " Please enter either 28, 50, or 280.\n")
    return False
  else:
    return True