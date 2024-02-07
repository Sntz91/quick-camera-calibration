import cv2
import numpy as np
import matplotlib.pyplot as plt
from lib.reference_points import ReferencePoints 

#images = ['DJI_0026', 'DJI_0029', 'DJI_0032', 'DJI_0035', 'DJI_0038', 'DJI_0045', 'DJI_0049', 'DJI_0053', 'DJI_0061', 'DJI_0066', 'DJI_0067', 'DJI_0078']
#colors = ['blue', 'tomato', 'green', 'yellow', 'darkslategray', 'gray', 'magenta', 'purple', 'aquamarine', 'beige', 'lavender', 'lightcyan']

images = ['DJI_0026', 'DJI_0045', 'DJI_0066']
colors = ['blue', 'tomato', 'green']

fname_img_tv = '/Users/tobias/data/5Safe/vup/homography_evaluation/data/top_view/DJI_0017.JPG'

img_tv = cv2.imread(fname_img_tv)
img_tv = cv2.cvtColor(img_tv, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(1, 1)
for imagename, colorname in zip(images, colors):
    fov_pts_tv = ReferencePoints.load(f'out/{imagename}.JPG/fov_pts_tv.txt')
    cam_pos = np.loadtxt(f'out/{imagename}.JPG/cam_pos.txt')

    ax = fov_pts_tv.plot_fov(img_tv, ax, color=colorname)
    ax.scatter(cam_pos[0], cam_pos[1], marker='*', s=100, color=colorname)
ax.axis('off')
plt.show()