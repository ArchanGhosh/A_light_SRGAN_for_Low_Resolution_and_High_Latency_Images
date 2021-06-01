import tensorflow as tf
import tensorflow.keras as keras
from keras import Input
from keras.layers import BatchNormalization, Activation, LeakyReLU, Add, Dense, Conv2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam

def residual_block(x):
  filters = [64, 64]
  kernel_size = 3
  strides = 1
  padding = "same"
  activation = "relu"
  momentum = 0.8

  res = Conv2D(filters=filters[0], kernel_size=kernel_size, padding=padding)(x)
  res = Activation(activation=activation)(res)
  res = BatchNormalization(momentum=momentum)(res)

  res = Conv2D(filters=filters[1], kernel_size=kernel_size, strides=strides, padding=padding)(res)
  res = BatchNormalization(momentum=momentum)(res)

  res = Add()([res, x])

  return res 