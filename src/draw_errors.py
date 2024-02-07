import cv2
import json
import matplotlib.pyplot as plt
from lib.reference_points import ReferencePoints 

# Load Images
imagename = 'DJI_0026'
fname_img_pv = f'/Users/tobias/data/5Safe/vup/homography_evaluation/data/perspective_views/{imagename}.JPG'
fname_img_tv = '/Users/tobias/data/5Safe/vup/homography_evaluation/data/top_view/DJI_0017.JPG'

img_pv = cv2.imread(fname_img_pv)
img_pv = cv2.cvtColor(img_pv, cv2.COLOR_BGR2RGB)
img_tv = cv2.imread(fname_img_tv)
img_tv = cv2.cvtColor(img_tv, cv2.COLOR_BGR2RGB)

fig, (ax1, ax2) = plt.subplots(1, 2)

# Load Results
ref_pts_pv = ReferencePoints.load(f'out/{imagename}.JPG/ref_pts_pv.txt')
val_pts_pv = ReferencePoints.load(f'out/{imagename}.JPG/val_pts_pv.txt')
val_pts_pv_transformed = ReferencePoints.load(f'out/{imagename}.JPG/val_pts_pv_transformed.txt')
val_pts_tv = ReferencePoints.load(f'out/{imagename}.JPG/val_pts_tv.txt')

d = {}
with open(f"out/{imagename}.JPG/summary.txt") as f:
    for line in f:
       (key, val) = line.split(';')
       d[key] = val

distances = json.loads(d['distances'].replace("\'", "\""))

ax1.imshow(img_pv)
ax2.imshow(img_tv)
for key, point in val_pts_pv.items():
    ax1.text(point.x, point.y, f"{distances[key]*100:.2f}")
    #ax1.text(point.x-300, point.y-100, f'{key}')

ax1 = ref_pts_pv.plot(img_pv, ax1, color='white')
ax1 = val_pts_pv.plot(img_pv, ax1, color='blue')
ax2 = val_pts_pv_transformed.plot(img_tv, ax2, color='red', marker='o', alpha=1.0, size=300)
ax2 = val_pts_tv.plot(img_tv, ax2, color='blue', marker='o', alpha=1.0, size=300)
ax1.axis('off')
ax2.axis('off')
plt.show()
