vgg = build_vgg()
vgg.compile(loss='mse', optimizer = common_optimizer, metrics=['accuracy'])
vgg.summary()

generator = build_generator()

discriminator = build_discriminator()
discriminator.trainable = True
discriminator.compile(loss='mse', optimizer= common_optimizer, metrics=['accuracy'])

adversarial_model = build_adversarial_model(generator, discriminator, vgg)