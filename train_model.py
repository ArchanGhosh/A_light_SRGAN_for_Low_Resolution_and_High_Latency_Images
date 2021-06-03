if not os.path.exists("./saved/"):
    os.makedirs("./saved/")
for epoch in range(epochs):

  d_history = []
  g_history = []

  high_resolution_images, low_resolution_images = sample_images(dir_data=dir_data, batch_size=batch_size, low_resolution_shape=low_resolution_shape, high_resolution_shape=high_resolution_shape)

  high_resolution_images = high_resolution_images/127.5 - 1.
  low_resolution_images = low_resolution_images/127.5 - 1.

  generated_high_resolution_images = generator.predict(low_resolution_images)

  real_labels = np.ones((batch_size, 8, 8, 1))
  fake_labels = np.zeros((batch_size, 8, 8, 1))

  d_loss_real = discriminator.train_on_batch(high_resolution_images, real_labels)
  d_loss_real = np.mean(d_loss_real)
  d_loss_fake = discriminator.train_on_batch(generated_high_resolution_images, fake_labels)
  d_loss_fake = np.mean(d_loss_fake)

  d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
  losses['d_history'].append(d_loss)


  high_resolution_images, low_resolution_images = sample_images(dir_data=dir_data, batch_size=batch_size, low_resolution_shape=low_resolution_shape, high_resolution_shape=high_resolution_shape)

  high_resolution_images = high_resolution_images/127.5 - 1.
  low_resolution_images = low_resolution_images/127.5 - 1.

  image_features = vgg.predict(high_resolution_images)

  gfaces_loss = adversarial_model.train_on_batch([low_resolution_images, high_resolution_images], [real_labels, image_features])

  losses['g_history'].append(0.5 * (gfaces_loss[1]))

  ps2 = calc_psnr2(high_resolution_images, generated_high_resolution_images)
  psnr2['psnr2_quality'].append(ps2)

  ss2 = calc_ssim2(high_resolution_images, generated_high_resolution_images)
  ssim2['ssim2_quality'].append(ss2)

  if epoch % 50 == 0:
    high_resolution_images, low_resolution_images = sample_images(dir_data=dir_data, batch_size=batch_size, low_resolution_shape=low_resolution_shape, high_resolution_shape=high_resolution_shape)
    
    high_resolution_images = high_resolution_images/127.5 - 1.
    low_resolution_images = low_resolution_images/127.5 - 1.

    generated_images = generator.predict_on_batch(low_resolution_images)

    for index, img in enumerate(generated_images):
      save_images(low_resolution_images[index], high_resolution_images[index], img, epoch, index)


plot_loss(losses)
plot_psnr2(psnr2)
plot_ssim2(ssim2)