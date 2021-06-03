valid_images_dir = 'VALID_DIR/*.png'

  all_images = glob.glob(valid_images_dir)

  valid_lr = []
  valid_hr = []

  for img in all_images:
    img1 = imread(img, as_gray=False, pilmode='RGB')
    img1 = img1.astype(np.float32)

    img1_high_resolution = imresize(img1, high_resolution_shape)
    img1_low_resolution = imresize(img1, low_resolution_shape)

    valid_hr.append(img1_high_resolution)
    valid_lr.append(img1_low_resolution)


valid_hr = np.array(valid_hr)
valid_lr = np.array(valid_lr)

valid_hr = valid_hr/127.5 - 1.
valid_lr = valid_lr/127.5 - 1.

valid_ge = generator.predict_on_batch(valid_lr)

for index, img in enumerate(valid_ge):
  plt.imshow(valid_hr[index], interpolation='nearest')
  plt.show()
  plt.imshow(valid_lr[index], interpolation='nearest')
  plt.show()
  plt.imshow(valid_ge[index], interpolation='nearest')
  plt.show()

