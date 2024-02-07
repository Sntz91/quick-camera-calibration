import numpy as np
import matplotlib.pyplot as plt
import cv2
from lib.reference_points import ReferencePoints

def save_experiment(
        output_dir,
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
):
    np.savetxt(f'{output_dir}/homography.txt', h)
    np.savetxt(f'{output_dir}/cam_pos.txt', cam_pos)
    ref_pts_pv.save(f'{output_dir}/ref_pts_pv.txt')
    ref_pts_tv.save(f'{output_dir}/ref_pts_tv.txt')
    val_pts_pv.save(f'{output_dir}/val_pts_pv.txt')
    val_pts_pv_transformed.save(f'{output_dir}/val_pts_pv_transformed.txt')
    val_pts_tv.save(f'{output_dir}/val_pts_tv.txt')
    fov_pts_pv.save(f'{output_dir}/fov_pts_pv.txt')
    fov_pts_tv.save(f'{output_dir}/fov_pts_tv.txt')

    fig1, fig2 = visualize_homography(h, ref_pts_pv, ref_pts_tv, img_pv, img_tv)
    fig1.subplots_adjust(0,0,1,1, wspace=0)
    fig1.savefig(f'{output_dir}/ref_pts.png', bbox_inches='tight', dpi=300)
    fig2.subplots_adjust(0,0,1,1)
    fig2.savefig(f'{output_dir}/homography_result.png', bbox_inches='tight',  dpi=300)
    fig3 = visualize_reprojection_error(img_tv, val_pts_tv, val_pts_pv_transformed)
    fig3.subplots_adjust(0,0,1,1)
    fig3.savefig(f'{output_dir}/reprojection_error.png', bbox_inches='tight', dpi=300)
    fig4 = visualize_fov(fov_pts_tv, img_tv, cam_pos)
    fig4.subplots_adjust(0,0,1,1)
    fig4.savefig(f'{output_dir}/fov.png', bbox_inches='tight', dpi=300)
    save_summary(f'{output_dir}/summary.txt', val_pts_tv, val_pts_pv_transformed, angle, altitude, nr_ref_pts, scaling_factor)
    plt.cla()



def save_summary(
        fname: str, 
        val_pts_tv: ReferencePoints, 
        val_pts_pv_transformed: ReferencePoints,
        angle: float,
        altitude: float,
        nr_ref_pts: int,
        scaling_factor = 59.3
    ) -> None:
    distances, mean_d, var_d, std_d, nr_pts = ReferencePoints.calc_distances(
        val_pts_tv, 
        val_pts_pv_transformed, 
        scaling_factor=scaling_factor
    )
    # If i can add names to the distances here, that would be awesome. TODO
    with open(fname, 'w') as file:
        file.write(
            'mean_distance;%f\nvariance;%f\nstd;%f\nsample_size;%d\ndistances;%s\nangle;%f\naltitude;%f\nnr_ref_pts;%d' \
            % (mean_d, var_d, std_d, nr_pts, distances, angle, altitude, nr_ref_pts)
    )

def visualize_homography(h, ref_pts_pv, ref_pts_tv, img_pv, img_tv):
    # Plot 1
    fig1, (ax1, ax2) = plt.subplots(1, 2)
    ax1 = ref_pts_pv.plot(img_pv, ax1)
    ax2 = ref_pts_tv.plot(img_tv, ax2)
    # Plot 2
    size = (img_tv.shape[1], img_tv.shape[0])
    output_img = cv2.warpPerspective(img_pv, h, size)
    fig2, ax3 = plt.subplots(1, 1)
    ax3.imshow(output_img)

    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    return fig1, fig2


def visualize_reprojection_error(img_tv, val_pts_tv, val_pts_pv_transformed):
    fig, ax = plt.subplots(1, 1)
    ax = val_pts_tv.plot(img_tv, ax, color='blue')
    ax = val_pts_pv_transformed.plot(img_tv, ax, marker='x')
    ax.axis('off')
    return fig


def visualize_fov(fov_pts_tv: ReferencePoints, tv_img: np.ndarray, cam_pos):
    fig, ax = plt.subplots(1, 1)
    ax = fov_pts_tv.plot_fov(tv_img, ax)
    ax.scatter(cam_pos[0], cam_pos[1], marker='x')
    ax.axis('off')
    return fig