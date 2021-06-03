import numpy as np
import pandas as pd
import random

from imageio import imread
from skimage.transform import resize as imresize
from copy import deepcopy
from tqdm import tqdm
import pprint
from PIL import Image




def sample_images(dir_data, batch_size, high_resolution_shape, low_resolution_shape):

  all_images = glob.glob(dir_data)

  images_batch = np.random.choice(all_images, size=batch_size)

  low_resolution_images = []
  high_resolution_images = []

  for img in images_batch:
    img1 = imread(img, as_gray=False, pilmode='RGB')
    img1 = img1.astype(np.float32)

    img1_high_resolution = imresize(img1, high_resolution_shape)
    img1_low_resolution = imresize(img1, low_resolution_shape)

    if np.random.random() < 0.5:
      img1_high_resolution = np.fliplr(img1_high_resolution)
      img1_low_resolution = np.fliplr(img1_low_resolution)

    high_resolution_images.append(img1_high_resolution)
    low_resolution_images.append(img1_low_resolution)

  return np.array(high_resolution_images), np.array(low_resolution_images)