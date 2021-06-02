import tensorflow as tf
import tensorflow.keras as keras
from keras import Input
from keras.layers import BatchNormalization, Activation, LeakyReLU, Add, Dense, Conv2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam

def build_generator():
  residual_blocks = 16
  momentum = 0.8

  input_shape = (32, 32, 3)

  input_layer = Input(shape=input_shape)

  gen1 = Conv2D(filters=64, kernel_size=9, strides=1, padding='same', activation='relu')(input_layer)

  res = residual_block(gen1)
  for i in range(residual_blocks - 1):
    res = residual_block(res)

  gen2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(res)
  gen2 = BatchNormalization(momentum=momentum)(gen2)

  gen3 = Add()([gen2, gen1])

  gen4 = UpSampling2D(size=2)(gen3)
  gen4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen4)
  gen4 = Activation('relu')(gen4)

  gen5 = UpSampling2D(size=2)(gen4)
  gen5 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen5)
  gen5 = Activation('relu')(gen5)

  gen6 = Conv2D(filters=3, kernel_size=9, strides=1, padding='same')(gen5)

  output = Activation('tanh')(gen6)

  model = Model(inputs=[input_layer], outputs=[output], name='generator')

  return model