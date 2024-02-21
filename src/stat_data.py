import cv2
import yaml
import numpy as np
import pandas as pd
from itertools import combinations
from lib.reference_points import ReferencePoints


def reference_points_experiment(img_name):
    # Load Config
    with open(f'conf/config_{img_name}.yaml', 'r') as file:
        cfg = yaml.safe_load(file)

    fname_val_pts_pv = cfg['perspective_view']['fname_val_pts']
    fname_ref_pts_pv = cfg['perspective_view']['fname_ref_pts']

    fname_val_pts_tv = cfg['top_view']['fname_val_pts']
    fname_ref_pts_tv = cfg['top_view']['fname_ref_pts']
    scaling_factor = cfg['top_view']['scaling_factor']

    selection_ref_pts = cfg['selection']['ref_pts']
    selection_val_pts = cfg['selection']['val_pts']

    val_pts_pv = ReferencePoints.load(fname_val_pts_pv)[selection_val_pts]
    val_pts_tv = ReferencePoints.load(fname_val_pts_tv)[selection_val_pts]

    data = []

    # For every combination
    for comb_len in range(4, len(selection_ref_pts)+1):
        print(f'start with {comb_len} nr of ref pts')
        combs = list(combinations(selection_ref_pts, comb_len))
        print(f'!!nr of experiments: {len(combs)}!!')

        # For every Experiment
        for i, comb in enumerate(combs):
            print(f'exp: {comb_len}, {i}')
            ref_pts_pv = ReferencePoints.load(fname_ref_pts_pv)[comb]
            ref_pts_tv = ReferencePoints.load(fname_ref_pts_tv)[comb]

            h = ReferencePoints.calc_homography_matrix(
                ref_pts_pv,
                ref_pts_tv
            )
            val_pts_pv_transformed = val_pts_pv.transform(h)
            distances, mean_d, var_d, std_d, nr_pts = ReferencePoints.calc_distances(
                val_pts_pv_transformed,
                val_pts_tv,
                scaling_factor
            )

            hull = cv2.convexHull(ref_pts_pv.get_numpy_arr().astype(int).reshape(-1, 1, 2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hull = np.squeeze(hull)
            _hull = np.concatenate((hull, [hull[0, :]]))

            for name, val_pt in val_pts_pv.items():
                quality = get_validation_point_quality(val_pt, hull)
                data.append({
                    'img': img_name,
                    'comb': comb_len,
                    'exp': i,
                    'val_pt': name,
                    'val_pt_x': val_pt.x,
                    'val_pt_y': val_pt.y,
                    'quality': quality,
                    'hull': hull,
                    'hull_len': len(hull),
                    'dist_to_hull': cv2.pointPolygonTest(_hull,
                                                         (val_pt.x, val_pt.y),
                                                         True),
                    'hull_area': cv2.contourArea(_hull),
                    'ref_pts': ref_pts_pv.get_numpy_arr(),
                    'n_ref_pts': len(ref_pts_pv),
                    'error': distances[name]*100
                })
    df = pd.DataFrame(data)
    return df


def get_validation_point_quality(val_pt, hull):
    if cv2.pointPolygonTest(hull, (val_pt.x, val_pt.y), False) == -1:
        return 'bad'
    return 'good'


def create_data(fname):
    df = reference_points_experiment(fname)
    df.to_pickle(f'out/statistics/{fname}_df.pickle')


def read_data(fname):
    df = pd.read_pickle(f'out/statistics/{fname}_df.pickle')
    return df


def filter_bad_hulls(df):
    return df[df.hull_len > 3]
