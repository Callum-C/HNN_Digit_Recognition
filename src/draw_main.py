"""

Utilise pygame to allow users to draw digits for the HNN to recognise.

"""

import numpy as np
import pygame
import math

"""

Initialise PyGame

"""

image_size = 50 # (image_size, image_size)

cellsize = 20 # Scalar to draw pixels / cells / neurons bigger

pygame.init()
surface = pygame.display.set_mode((600, 600))
surface.fill((255,255,255))
pygame.display.set_caption("  ")

cells = np.zeros((image_size, image_size))

def main_loop(cells):
  """
  
  Main loop for drawing to the canvas.
  - Left click to draw on the canvas.
  - Right click to clear the canvas.

  """

  running = True

  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
        pygame.quit()
        return
    
    # Get position of mouse and state of buttons pressed
    cur = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()

    # Clear surface when right click pressed
    if click[2]:
      surface.fill((255,255,255))
      cells = np.zeros((image_size, image_size))

    update_cells(cur, click, cells)
    
    # Draw user input to display
    # draw(cur, click) # - Draw exact mouse position on canvas in blue
    draw_cells(cells) # - Draw cells to canvas in black
      
    
def update_cells(cur, click, cells):
  """
  
  Update stored information in cells while left click depressed.

  """
  if click[0]:
    c = math.floor(cur[0] / cellsize)
    r = math.floor(cur[1] / cellsize)
    cells[c][r] = 1


def draw(cur, click):
  """
  
  Draw a rectangle of size (cellsize, cellsize) at the mouse's actual position.

  """
  if click[0]:
    pygame.draw.rect(surface, (0,0,255), (cur[0], cur[1], cellsize, cellsize))

  pygame.display.update()


def draw_cells(cells):
  """
  
  Turn the stored information of where to draw into a drawing.
  Much more "pixelated" version of draw()
  
  """

  for i,row in enumerate(cells):
    for j,cell in enumerate(row):
      if cell == 1:
        pygame.draw.rect(
          surface, (0,0,0), (i*cellsize, j*cellsize, cellsize, cellsize)
        )
  
  pygame.display.update()

main_loop(cells)