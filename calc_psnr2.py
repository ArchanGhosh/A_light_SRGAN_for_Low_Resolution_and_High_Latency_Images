def calc_psnr2(original_image, generated_image):
  original_image = tf.convert_to_tensor(original_image, dtype=tf.float32)
  generated_image = tf.convert_to_tensor(generated_image, dtype=tf.float32)
  psnr2 = tf.image.psnr(original_image, generated_image, max_val=1.0)

  return tf.math.reduce_mean(psnr2, axis=None, keepdims=False, name=None)