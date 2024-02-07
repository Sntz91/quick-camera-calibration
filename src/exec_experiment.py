import yaml
import cv2
import os
from lib.reference_points import ReferencePoints 
from lib.save_experiment import save_experiment

def calc_homography_and_reprojection_error(cfg_name):
    # --------------------------------------------------------------------------
    # 0. Parse Config 
    # --------------------------------------------------------------------------
    with open(cfg_name, 'r') as file:
        cfg = yaml.safe_load(file)

    fname_img_pv = cfg['perspective_view']['fname_img']
    fname_val_pts_pv = cfg['perspective_view']['fname_val_pts']
    fname_ref_pts_pv = cfg['perspective_view']['fname_ref_pts']
    fname_fov_pts_pv = cfg['perspective_view']['fname_fov_pts']
    fname_K = cfg['perspective_view']['fname_K']
    angle = cfg['perspective_view']['angle']
    altitude = cfg['perspective_view']['altitude']

    fname_img_tv = cfg['top_view']['fname_img']
    fname_val_pts_tv = cfg['top_view']['fname_val_pts']
    fname_ref_pts_tv = cfg['top_view']['fname_ref_pts']
    scaling_factor = cfg['top_view']['scaling_factor']

    selection_ref_pts = cfg['selection']['ref_pts']
    selection_val_pts = cfg['selection']['val_pts']
    dirname_result = cfg['output']['dir']

    ref_pts_pv = ReferencePoints.load(fname_ref_pts_pv)[selection_ref_pts]
    ref_pts_tv = ReferencePoints.load(fname_ref_pts_tv)[selection_ref_pts]
    val_pts_pv = ReferencePoints.load(fname_val_pts_pv)[selection_val_pts]
    val_pts_tv = ReferencePoints.load(fname_val_pts_tv)[selection_val_pts]
    fov_pts_pv = ReferencePoints.load(fname_fov_pts_pv)
    nr_ref_pts = len(ref_pts_pv)

    img_pv = cv2.imread(fname_img_pv)
    img_tv = cv2.imread(fname_img_tv)
    img_pv = cv2.cvtColor(img_pv, cv2.COLOR_BGR2RGB)
    img_tv = cv2.cvtColor(img_tv, cv2.COLOR_BGR2RGB)


    # --------------------------------------------------------------------------
    # 1. Calc Homography
    # --------------------------------------------------------------------------
    h = ReferencePoints.calc_homography_matrix(
        ref_pts_pv,
        ref_pts_tv
    )


    # --------------------------------------------------------------------------
    # 2. Calc Reprojection Error
    # --------------------------------------------------------------------------
    val_pts_pv_transformed = val_pts_pv.transform(h)
    distances, mean_d, var_d, std_d, nr_pts = ReferencePoints.calc_distances(
        val_pts_pv_transformed,
        val_pts_tv,
        scaling_factor
    )
    print(f'Error: {mean_d: .2f} m', f'{mean_d*100: .2f} cm') 


    # --------------------------------------------------------------------------
    # 3. Calc FoV
    # --------------------------------------------------------------------------
    fov_pts_tv = fov_pts_pv.calc_fov(h)

    # Camera Position
    K, dist = ReferencePoints.load_K(fname_K)
    rvec, tvec = ReferencePoints.get_rotation_and_translation_vector(ref_pts_tv, ref_pts_pv, K, dist)
    rotM = ReferencePoints.get_rotation_matrix(rvec)
    cam_pos = ReferencePoints.get_camera_position(rotM, tvec)

    # --------------------------------------------------------------------------
    # 4. Save Experiment
    # --------------------------------------------------------------------------
    os.mkdir(dirname_result)
    save_experiment(
        dirname_result,
        h,
        ref_pts_pv,
        ref_pts_tv,
        val_pts_pv,
        val_pts_tv,
        val_pts_pv_transformed,
        fov_pts_pv,
        fov_pts_tv,
        img_pv,
        img_tv,
        cam_pos,
        scaling_factor,
        angle,
        altitude,
        nr_ref_pts
    )

if __name__ == '__main__':
    calc_homography_and_reprojection_error('conf/config_DJI_0026.yaml')
    calc_homography_and_reprojection_error('conf/config_DJI_0029.yaml')
    calc_homography_and_reprojection_error('conf/config_DJI_0032.yaml')
    calc_homography_and_reprojection_error('conf/config_DJI_0035.yaml')
    calc_homography_and_reprojection_error('conf/config_DJI_0038.yaml')
    calc_homography_and_reprojection_error('conf/config_DJI_0045.yaml')
    calc_homography_and_reprojection_error('conf/config_DJI_0049.yaml')
    calc_homography_and_reprojection_error('conf/config_DJI_0053.yaml')
    calc_homography_and_reprojection_error('conf/config_DJI_0061.yaml')
    calc_homography_and_reprojection_error('conf/config_DJI_0066.yaml')
    calc_homography_and_reprojection_error('conf/config_DJI_0067.yaml')
    calc_homography_and_reprojection_error('conf/config_DJI_0078.yaml')
