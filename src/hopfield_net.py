import numpy as np

class hopfieldNet: 
  """
  Hopfield Neural Network class.
  """

  def __init__(self, input, images):
    """
    Initialize a Hopfield Neural Network object.
    Set network variables and memory.
    """

    # Patterns for network training / retrieval
    self.memory = np.array(input)

    self.n = self.memory.shape[1] # Image size: (28x28)

    # Construct network
    # - Initialize State vector
    img_index = np.random.randint(0, self.memory.shape[0])
    img = images[img_index]
    
    (row, col) = (28, 28)

    number_of_pixels = np.random.randint(50, 200)
    for i in range(number_of_pixels):
      y_coord = np.random.randint(0, row-1)
      x_coord = np.random.randint(0, col-1)

      img[y_coord][x_coord] = 1.0

    number_of_pixels = np.random.randint(50, 200)
    for i in range(number_of_pixels):
      y_coord = np.random.randint(0, row-1)
      x_coord = np.random.randint(0, col-1)

      img[y_coord][x_coord] = -1.0

    self.state = np.array(img).flatten()

    # - Initialize Weight vector
    self.weights = np.zeros((self.n, self.n))

    # - Initialize Energies
    self.energies = [] # container for tracking of energy

  def network_learning(self): 
    """
    Learn the pattern / patterns.
    """

    # hebbian learning
    self.weights = (1 / self.memory.shape[0]) * self.memory.T @ self.memory
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