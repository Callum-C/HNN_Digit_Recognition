import numpy as np
import matplotlib.pyplot as plt
import cv2
import pygame

from hopfield_net import hopfieldNet


fname = 'Digits/6.png'
img = cv2.imread(fname, 0)
img_norm = cv2.normalize(img, None, -1, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
img_flat = np.array(img_norm).flatten()

# Snag a memory from computer brain
memories = np.array([img_flat])

# Initalize Hopfield Network
net = hopfieldNet(memories)
net.train()

# Initalize pygame
cellsize = 20
pygame.init()
surface = pygame.display.set_mode((28*cellsize, 28*cellsize))
pygame.display.set_caption("  ")

pause_at_start = True
running = True

# Main Animation Loop
while running:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False

      # TODO Draw Graphs

      pygame.quit()
  
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


