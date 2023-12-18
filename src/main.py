import numpy as np
import pygame
import matplotlib.pyplot as plt

from hopfield_net import hopfieldNet
from images import *
from graphs import plot_graphs

"""

Simulation settings

"""

# (image_size, image-size) - 28, 50 or 280
image_size = 50  

# Number of images to learn 1 - 10
# If single_digit method used, this is the digit to learn
number_of_memories = 3 

# Of those images learned, which one should be tested
# - Number should be positive integer but lower than number of memories
recreate_memory = 1  

# Noise to add to the recreated memory / test image - integer
amount_of_noise = int((image_size * image_size) / 4)  

# Should simulation pause at the start to show starting_state?
pause_at_start = False

# Show energy and weight graphs after simulation? 
show_graphs = False 

"""

Main Code

"""

# Memory selection method

memories = read_images(number_of_memories, image_size)
#memories = read_odd_digits(number_of_memories, image_size)
#memories = read_single_digit(number_of_memories, image_size)

print("Memories shape: {}".format(memories.shape))

# Set starting state
starting_state = add_rng_noise(memories[recreate_memory], amount_of_noise)

# Initalise Hopfield Network
net = hopfieldNet(memories, starting_state)
net.train()

# Initalise pygame
cellsize = 20
pygame.init()
surface = pygame.display.set_mode((net.sqrt_n*cellsize, net.sqrt_n*cellsize))
pygame.display.set_caption("  ")

def main_loop(pause_at_start=True):
  """
  Randomly update nodes' states and animate change using pygame.
  """
  running = True

  # Main Animation Loop
  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False

        if show_graphs:
          plot_graphs(net)

        pygame.quit()
        return
    
    cells = net.states.reshape(net.sqrt_n, net.sqrt_n).T

    surface.fill((211, 211, 211))

    # Loop through network states and update colours for each cell
    for r, c in np.ndindex(cells.shape):
      if cells[r, c] == -1:
        col = (0, 0, 0)

      elif cells[r, c] == 1:
        col = (255, 255, 255)

      pygame.draw.rect(
        surface, col, (r*cellsize, c*cellsize, cellsize, cellsize)
      )

    # Update network states
    net.update_states(32)
    net.compute_energy()
    pygame.display.update()
    if pause_at_start:
      pygame.time.wait(1500)
      pause_at_start = False

main_loop(pause_at_start)
plt.show()

