import cv2
from lib.reference_points import ReferencePoints

def main(
    fname_pv_img: str,
    fname_tv_img: str,
    dirname_output: str
) -> None:
    pv_img = cv2.imread(fname_pv_img)
    tv_img = cv2.imread(fname_tv_img)
    pv_img = cv2.cvtColor(pv_img, cv2.COLOR_BGR2RGB)
    tv_img = cv2.cvtColor(tv_img, cv2.COLOR_BGR2RGB)

    ReferencePoints.set_reference_pts(pv_img, tv_img, dirname_output)


if __name__ == '__main__':
    fname_pv_img = '/Users/tobias/data/5Safe/vup/homography_evaluation/data/perspective_views/DJI_0026.JPG'
    fname_tv_img = '/Users/tobias/data/5Safe/vup/homography_evaluation/data/top_view/DJI_0017.JPG'
    dirname_output = 'out'

    main(fname_pv_img, fname_tv_img, dirname_output)
