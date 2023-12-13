import numpy as np
import cv2
import math

class hopfieldNet:
  """
  Hopfield Neural Network for Digit Recognition.
  """

  def __init__(self, input):
    """
    Construct a Hopfield Neural Network object.
    Set network variables and memories.
    """

    # Patterns for network to remember
    self.memories = np.array(input)

    # m = number of memories stored
    self.m = self.memories.shape[0]

    # n = number of neurons in a row
    # - Square grid so total number of nuerons is n^2
    self.n = self.memories.shape[1]
    self.sqrt_n = int(math.sqrt(self.n))
    print("Neurons: {}".format(self.n))

    # Construct network
    self.states = np.random.randint(0, 2, (self.n, 1))
    self.states = cv2.normalize(self.states, None, -1, 1.0, cv2.NORM_MINMAX, 
                                dtype=cv2.CV_64F)
    self.weights = np.zeros((self.n, self.n))
    self.energies = []

  def train(self):
    """
    Learn the memories / train the network.
    """

    self.weights = (1 / self.m) * self.memories.T @ self.memories
    np.fill_diagonal(self.weights, 0)

  def update_states(self, n_updates):
    """
    Pick random neurons n_updates times and update their state.
    """

    for neuron in range(n_updates):
      self.rand_index = np.random.randint(0, self.n)

      self.index_activation = np.dot(self.weights[self.rand_index, :],
                                     self.states)
      
      if self.index_activation < 0:
        self.states[self.rand_index] = -1
      else:
        self.states[self.rand_index] = 1

  def compute_energy(self):
    """
    Compute the total energy of the network
    """

    self.energy = -0.5 * np.dot(np.dot(self.states.T, self.weights), self.states)
    self.energies.append(self.energy)

