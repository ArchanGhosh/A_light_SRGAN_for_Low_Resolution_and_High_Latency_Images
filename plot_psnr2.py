def plot_psnr2(psnr2):
  psnr2_means = psnr2['psnr2_quality']
  plt.figure(figsize=(10,8))
  plt.plot(psnr2_means, label='PSNR Quality')
  plt.xlabel('Epochs')
  plt.ylabel('PSNR')
  plt.legend()
  plt.show()