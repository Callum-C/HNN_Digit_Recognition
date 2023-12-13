import numpy as np
import matplotlib.pyplot as plt
import cv2
import pygame

from hopfield_net import hopfieldNet
from images import read_images, read_single_digit

#memories = read_images()
memories = read_single_digit(4)

print("Memories shape: {}".format(memories.shape))

# Initalize Hopfield Network
net = hopfieldNet(memories)
net.train()

# Initalize pygame
cellsize = 20
pygame.init()
surface = pygame.display.set_mode((28*cellsize, 28*cellsize))
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

        # TODO Draw Graphs

        pygame.quit()
        return
    
    cells = net.states.reshape(28, 28).T

    surface.fill((211, 211, 211))

    # Loop through network states and update colours for each cell
    for r, c in np.ndindex(cells.shape):
      if cells[r, c] == -1:
        col = (0, 0, 0)

      elif cells[r, c] == 1:
        col = (255, 255, 255)

      pygame.draw.rect(surface, col, (r*cellsize, c*cellsize, cellsize, cellsize))

    # Update network states
    net.update_states(16)
    net.compute_energy()
    pygame.display.update()
    if pause_at_start:
      pygame.time.wait(2000)
      pause_at_start = False

main_loop()
print(net.energies)
print(net.weights)

