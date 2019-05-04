import imageio
import os

root = './cnimgs/'
def create_gif(image_list, gif_name):
    frames = []
    for image_name in image_list:
        image_name = os.path.join(root,str(image_name)+'.jpg')
        frames.append(imageio.imread(image_name))
        # Save them as frames into a gif
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.1)
    return
buf = []
def find_all_png():
    for file in os.listdir(root):
        buf.append(int(file.split('.')[0]))
    return buf
if __name__ == '__main__':
    buff = sorted(find_all_png())
    create_gif(buff, 'cn.gif')