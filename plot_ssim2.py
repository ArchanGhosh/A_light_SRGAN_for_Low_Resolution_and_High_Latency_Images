def plot_ssim2(ssim2):
  ssim2_means = ssim2['ssim2_quality']
  plt.figure(figsize=(10,8))
  plt.plot(ssim2_means, label='SSIM Quanlity')
  plt.xlabel('Epochs')
  plt.ylabel('SSIM')
  plt.legend()
  plt.show()