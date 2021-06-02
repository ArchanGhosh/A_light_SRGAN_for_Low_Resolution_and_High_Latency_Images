def calc_ssim2(original_image, generated_image):
  original_image = tf.convert_to_tensor(original_image, dtype=tf.float32)
  generated_image = tf.convert_to_tensor(generated_image, dtype=tf.float32)
  ssim2 = tf.image.ssim(original_image, generated_image, max_val = 1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
  return tf.math.reduce_mean(ssim2, axis=None, keepdims=False, name=None)