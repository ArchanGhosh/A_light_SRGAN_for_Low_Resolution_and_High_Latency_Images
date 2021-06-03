frames = []
imgs = glob.glob("saved/*.png")
imgs.sort()
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

 

frames[0].save('SRGAN_32-128.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=700, loop=0)