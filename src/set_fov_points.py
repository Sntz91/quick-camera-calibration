import cv2
from lib.reference_points import ReferencePoints

def main(
    fname_pv_img: str,
    fname_output: str
) -> None:
    pv_img = cv2.imread(fname_pv_img)
    pv_img = cv2.cvtColor(pv_img, cv2.COLOR_BGR2RGB)

    border_pts = ReferencePoints()
    border_pts.set_fov_pts(
        pv_img, 
        fname_output
    )
    

if __name__ == '__main__':
    img_nr = '0078'
    fname_pv_img = f'/Users/tobias/data/5Safe/vup/homography_evaluation/data/perspective_views/DJI_{img_nr}.JPG'
    fname_output = f'/Users/tobias/data/5Safe/vup/homography_evaluation/annotations/DJI_{img_nr}_fov_pts.txt'

    main(fname_pv_img, fname_output)