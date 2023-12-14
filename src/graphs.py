import matplotlib.pyplot as plt
import numpy as np

def plot_graphs(hnn):
  """
  Plot and display all graphs from the simulation.

  Params
  ------
  hnn: hopfieldNet - The hopfield network 
  """

  # Plot Weights Graph
  plt.figure("Weights", figsize=(10, 7))
  plt.imshow(hnn.weights, cmap='RdPu')
  plt.xlabel(
    "Each row / column represents a neuron, and each square a connection"
  )
  plt.title("Neurons and Their Connections", fontsize=15)
  plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

  # Plot Energies
  plt.figure("Energy", figsize=(10, 7))
  x = np.arange(len(hnn.energies))
  plt.scatter(x, np.array(hnn.energies), s=1, color='red')
  plt.xlabel("Generation")
  plt.ylabel("Energy")
  plt.title("Network Energy over Successive Generations", fontsize=15)
  plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])