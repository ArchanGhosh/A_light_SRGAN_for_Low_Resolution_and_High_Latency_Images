import tensorflow as tf
import tensorflow.keras as keras
from keras import Input
from keras.layers import BatchNormalization, Activation, LeakyReLU, Add, Dense, Conv2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam

def build_discriminator():

  leakyrelu_alpha = 0.2
  momentum = 0.8

  input_shape = (128, 128, 3)

  input_layer = Input(shape=input_shape)

  dis1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(input_layer)
  dis1 = LeakyReLU(alpha=leakyrelu_alpha)(dis1)

  dis2 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(dis1)
  dis2 = LeakyReLU(alpha=leakyrelu_alpha)(dis2)
  dis2 = BatchNormalization(momentum=momentum)(dis2)

  dis3 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(dis2)
  dis3 = LeakyReLU(alpha=leakyrelu_alpha)(dis3)
  dis3 = BatchNormalization(momentum=momentum)(dis3)

  dis4 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(dis3)
  dis4 = LeakyReLU(alpha=leakyrelu_alpha)(dis4)
  dis4 = BatchNormalization(momentum=momentum)(dis4)

  dis5 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(dis4)
  dis5 = LeakyReLU(alpha=leakyrelu_alpha)(dis5)
  dis5 = BatchNormalization(momentum=momentum)(dis5)

  dis6 = Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(dis5)
  dis6 = LeakyReLU(alpha=leakyrelu_alpha)(dis6)
  dis6 = BatchNormalization(momentum=momentum)(dis6)

  dis7 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(dis6)
  dis7 = LeakyReLU(alpha=leakyrelu_alpha)(dis7)
  dis7 = BatchNormalization(momentum=momentum)(dis7)

  dis8 = Conv2D(filters=512, kernel_size=3, strides=2, padding='same')(dis7)
  dis8 = LeakyReLU(alpha=leakyrelu_alpha)(dis8)
  dis8 = BatchNormalization(momentum=momentum)(dis8)

  dis9 = Dense(units=1024)(dis8)
  dis9 = LeakyReLU(alpha=0.2)(dis9)

  output = Dense(units=1, activation='sigmoid')(dis9)

  model = Model(inputs=[input_layer], outputs=output, name='discriminator')

  return model
