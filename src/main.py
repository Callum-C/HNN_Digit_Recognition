import numpy as np
import matplotlib.pyplot as plt
import cv2
import pygame

from hopfield_net import hopfieldNet

def Test_Hopfield():
  """
  Test out the Hopfield_Network object on some MNIST data.
  """

  fname = 'Digits/2.png'
  img = cv2.imread(fname, 0)
  img_norm = cv2.normalize(img, None, -1, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

  # Snag a memory from computer brain
  memories = np.array(img_norm)
  
  # Initalize Hopfield object
  H_Net = hopfieldNet(memories)
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
        plt.figure("Energy",figsize=(10,7))
        x = np.arange(len(H_Net.energies))
        plt.scatter(x,np.array(H_Net.energies),s=1,color='red')
        plt.xlabel("Generation")
        plt.ylabel("Energy")
        plt.title("Network Energy over Successive Generations",fontsize=15)
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

        # quit pygame
        pygame.quit()
        return

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
    # pygame.time.wait(50)


Test_Hopfield()
plt.show()

      




