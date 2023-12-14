import numpy as np
import pygame
import matplotlib.pyplot as plt

from hopfield_net import hopfieldNet
from images import read_images, read_single_digit, add_rng_noise
from graphs import plot_graphs

memories = read_images(2)
# memories = read_single_digit(6, 50)

starting_state = add_rng_noise(memories[1], 1000)

# Initalize Hopfield Network
net = hopfieldNet(memories, starting_state)
net.train()

# Initalize pygame
cellsize = 20
pygame.init()
surface = pygame.display.set_mode((net.sqrt_n*cellsize, net.sqrt_n*cellsize))
pygame.display.set_caption("  ")

def main_loop():
  """
  Randomly update nodes' states and animate change using pygame.
  """
  pause_at_start = True
  running = True

  # Main Animation Loop
  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False

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

      pygame.draw.rect(surface, col, (r*cellsize, c*cellsize, cellsize, cellsize))

    # Update network states
    net.update_states(32)
    net.compute_energy()
    pygame.display.update()
    if pause_at_start:
      pygame.time.wait(1500)
      pause_at_start = False

main_loop()
plt.show()

