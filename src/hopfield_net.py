import numpy as np
import cv2
import math

class hopfieldNet:
  """
  Hopfield Neural Network for Digit Recognition.
  """

  def __init__(self, memories, starting_state=[]):
    """
    Construct a Hopfield Neural Network object.
    Set network variables and memories.
    """

    # Patterns for network to remember
    self.memories = np.array(memories)

    # m = number of memories stored
    self.m = self.memories.shape[0]

    # n = total number of neurons
    # sqrt_n = number of neurons in a row when image is not flattened
    self.n = self.memories.shape[1]
    self.sqrt_n = int(math.sqrt(self.n))

    # Construct network
    if len(starting_state) == 0:
      self.states = np.random.randint(0, 2, (self.n, 1))
    else:
      self.states = starting_state

    self.states = cv2.normalize(self.states, None, -1, 1.0, cv2.NORM_MINMAX, 
                                dtype=cv2.CV_64F)
    self.weights = np.zeros((self.n, self.n))
    self.energies = []


  def train_hebbian(self):
    """
    Learn the memories / train the network.
    - Uses Hebbian learning
    """

    self.weights = (1 / self.m) * self.memories.T @ self.memories
    np.fill_diagonal(self.weights, 0)


  def train_storkey(self):
    """
    Learn the memories / train the network.
    - Uses the Storkey training method
    """

    for memory in self.memories:
      old_weights = self.weights.copy()
      hebbian_term = np.outer(memory, memory.T)

      net_inputs = old_weights.dot(memory)
      net_inputs = np.tile(net_inputs, (self.n, 1))

      h_i = np.diagonal(old_weights) * memory
      h_i = h_i[:, np.newaxis]

      h_j = old_weights * memory
      np.fill_diagonal(h_j, 0)

      hij = net_inputs - h_i - h_j

      post_synaptic = hij * memory
      pre_synaptic = hij.T * memory[:, np.newaxis]

      self.weights = old_weights + (1./self.n) * (hebbian_term - pre_synaptic - post_synaptic)


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