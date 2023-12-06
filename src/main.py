import numpy as np
import matplotlib.pyplot as plt
import requests, gzip, os, hashlib
import pygame

from hopfield_net import hopfieldNet

# Fetch MNIST dataset from the ~SOURCE~
def fetch_MNIST(url):
  fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
  if os.path.isfile(fp):
    dat = f.read()
  else:
    with open(fp, "wb") as f:
      dat = requests.get(url).content
      f.write(dat)

  return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()

def MNIST_Hopfield():
  """
  Test out the Hopfield_Network object on some MNIST data.
  Fetch MNIST dataset for some random memory downloads.
  """

  X = fetch_MNIST("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
        )[0x10:].reshape((-1,784))
  
  # Convert to binary
  X_binary = np.where(X>20, 1, -1)

  # Snag a memory from computer brain
  memories_list = np.array([X_binary[np.random.randint(len(X))]])

  # Initalize Hopfield object
  H_Net = hopfieldNet(memories_list)
  H_Net.network_learning()

  # Draw it all out, updating board each update iteration
  cellsize = 20

  # Initialize pygame
  pygame.init()
  # set dimensions of board and cellsize - 28 x 28 ~ display surface
  surface = pygame.display.set_mode((28*cellsize, 28*cellsize))
  pygame.display.set_caption("  ")

  # Kill pygame if user exits window
  Running = True
  # Main animation loop
  while Running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        Running = False

        # Plot weights matrix
        plt.figure("weights", figsize=(10,7))
        plt.imshow(H_Net.weights, cmap='RdPu')
        plt.xlabel("Each row/column represents a neuron, each square a connection")

        plt.title(" 4096 Neurons - 16,777,216 unique connections", fontsize=15)
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

        # Plot energies
        plt.figure("Energy", figsize=(10,7))
        x = np.arange(len(H_Net.energies), s=1, color='red')
        plt.xlabel("Generation")
        plt.ylabel("Energy")
        plt.title("Network Energy over Successive Generations", fontsize=15)
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

        # quit pygame
        pygame.quit()

    cells = H_Net.state.reshape(28,28).T

    # Fills surface with colour
    surface.fill((211, 211, 211))

    # Loop through network state array and update colours for each cell
    for r, c in np.ndindex(cells.shape): # Iterates through all cells in cells matrix
      if cells[r, c] == -1:
        col = (135, 206, 250)

      elif cells[r, c] == 1:
        col = (0, 0, 128)

      else:
        col = (255, 140, 0)
      
      pygame.draw.rect(surface, col, (r*cellsize, c*cellsize, 
                                      cellsize, cellsize))
      
    # Update network state
    H_Net.update_network_state(16)
    H_Net.compute_energy()
    pygame.display.update() # Updates display from new .draw in update function
    pygame.time.wait(50)


MNIST_Hopfield()
plt.show()

      




