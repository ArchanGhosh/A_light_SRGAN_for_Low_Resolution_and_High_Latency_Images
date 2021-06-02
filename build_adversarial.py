def build_adversarial_model(generator, discriminator, vgg):

  input_high_resolution = Input(shape=high_resolution_shape)

  input_low_resolution = Input(shape=low_resolution_shape)

  generated_high_resolution_images = generator(input_low_resolution)

  features = vgg(generated_high_resolution_images)

  discriminator.trainable = False
  discriminator.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])

  probs = discriminator(generated_high_resolution_images)

  adversarial_model = Model([input_low_resolution, input_high_resolution], [probs, features])
  adversarial_model.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1], optimizer=common_optimizer)

  return adversarial_model