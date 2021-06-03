import numpy as np
import pandas as pd
import random

from imageio import imread
from skimage.transform import resize as imresize
from copy import deepcopy
from tqdm import tqdm
import pprint
from PIL import Image


def save_images(low_resolution_image, original_image, generated_image, epoch, index):
  
  fig = plt.figure(figsize=(10,10))

  ax = fig.add_subplot(1, 3, 1)
  ax.imshow(original_image)
  ax.axis("off")
  ax.set_title("ORIGINAL")

  ax = fig.add_subplot(1, 3, 2)
  ax.imshow(low_resolution_image)
  ax.axis("off")
  ax.set_title("LOW_RESOLUTION")

  ax = fig.add_subplot(1, 3, 3)
  ax.imshow(generated_image)
  ax.axis("off")
  ax.set_title("generated")

  plt.savefig("./saved/img_{}_{}.png".format(epoch, index))