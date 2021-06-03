def plot_loss(losses):

  d_loss = losses['d_history']
  g_loss = losses['g_history']

  plt.figure(figsize=(10, 8))
  plt.plot(d_loss, label="Discriminator Loss")
  plt.plot(g_loss, label="Generator Loss")

  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()