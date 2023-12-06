import numpy as np

class hopfieldNet: 
  """
  Hopfield Neural Network class.
  """

  def __init__(self, input):
    """
    Initialize a Hopfield Neural Network object.
    Set network variables and memory.
    """

    # Patterns for network training / retrieval
    self.memory = np.array([np.array(input).flatten()])

    self.n = self.memory.shape[1] # Image size: (28x28)

    # Construct network
    self.state = np.random.randint(0, 2, (self.n, 1)) # state vector
    self.weights = np.zeros((self.n, self.n)) # weights vector
    self.energies = [] # container for tracking of energy

  def network_learning(self): 
    """
    Learn the pattern / patterns.
    """

    # hebbian learning
    self.weights = (1 / self.memory.shape[0]) * self.memory.T @ self.memory # 1 should be (1 / Number of patterns)
    np.fill_diagonal(self.weights, 0)

  
  def update_network_state(self, n_update):
    """
    Update Network.
    """

    # update n neurons randomly
    for neuron in range(n_update): 
      # pick random neuron in the state vector
      self.rand_index = np.random.randint(0, self.n) 
      # Compute activation for randomly indexed neuron
      self.index_activation = np.dot(self.weights[self.rand_index, :], 
                                     self.state)
      print(self.index_activation)
      # threshold function for binary state change
      if self.index_activation < 0:
        self.state[self.rand_index] = -1
      else:
        self.state[self.rand_index] = 1

  def compute_energy(self): 
    """
    Compute energy.
    """

    self.energy = -0.5 * np.dot(np.dot(self.state.T, self.weights), self.state)
    self.energies.append(self.energy)