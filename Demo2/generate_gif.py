import imageio
import os

def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread('result_image/' + image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return

def main():
    path = 'result_image/'
    image_list = os.listdir(path)
    image_list.sort(key = lambda x:int(x[:-4]))
    duration = 0.35
    gif_name = path + 'fakes.gif'
    create_gif(image_list, gif_name, duration)

if __name__ == '__main__':
    main()