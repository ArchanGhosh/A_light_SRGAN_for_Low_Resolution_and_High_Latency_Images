import tensorflow as tf
import tensorflow.keras as keras

from keras.applications import VGG19

def build_vgg():
  input_layer = Input(shape=(128, 128, 3))

  vgg = VGG19(weights="imagenet", include_top=False, input_tensor=input_layer)
  vgg.trainable = False 

  outputs = vgg.layers[9].output


  model = Model(inputs=[input_layer], outputs=[outputs])

  return model