from dataclasses import dataclass
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json

@dataclass
class Point:
    """ item of reference point container """
    x: int
    y: int


class ReferencePoints(dict):
    """ Container Class for reference points """
    def __init__(self):
        super().__init__()

    def __iter__(self):
        return iter(self)
    
    def __setitem__(self, item, value):
        assert type(value) == Point
        super().__setitem__(item, value)

    def __getitem__(self, idxs):
        if not type(idxs) in [tuple, list]:
            return super().__getitem__(idxs)
        out = ReferencePoints()
        for idx in idxs:
            out[idx] = self[idx]
        return out
    
    def get_numpy_arr(self):
        out = []
        for name, point in self.items():
            out.append([point.x, point.y])
        return np.array(out)

    def _as_txt(self):
        out = ''
        sorted_ = self.sort()
        for name, point in sorted_.items():
            out += f'{name} {point.x} {point.y}\n'
        return out

    def save(self, filename):
        with open(filename, 'w') as file:
            file.write(self._as_txt())

    @staticmethod
    def load(filename):
        out = ReferencePoints()
        with open(filename, 'r') as file:
            for line in file:
                name, x, y = line.split()
                out[name] = Point(float(x), float(y))
        return out.sort()
    
    @staticmethod
    def calc_l2_error(pts1, pts2, scaling_factor=1):
        err = np.linalg.norm(
            pts1.get_numpy_arr() - pts2.get_numpy_arr()
        )
        return err / scaling_factor
    
    @staticmethod
    def calc_distances(pts1, pts2, scaling_factor=1):
        # TODO Be More safe with names.., sort after number in string...
        names = pts1.keys()

        val_pts_gt = pts1.get_numpy_arr()
        val_pts_pred = pts2.get_numpy_arr()
        distances = np.sqrt(
            np.sum((val_pts_pred - val_pts_gt)**2, axis=1)
        ) / scaling_factor
        mean_distances = np.mean(distances)
        var_distances = np.var(distances)
        #print('variance:', var_distances)
        std_distances = np.std(distances)
        #print('std:', std_distances)
        sample_size = len(distances)
        distances_w_name = {name: distance for name, distance in zip(names, distances)}

        return distances_w_name, mean_distances, var_distances, std_distances, sample_size

    def sort(self):
        out = ReferencePoints()
        for idx in sorted(self.keys()):
            out[idx] = self[idx]
        return out
    
    def transform(self, h):
        out = ReferencePoints()
        names = self.keys()
        object_pts = self.get_numpy_arr()
        homogeneous_c = np.ones((object_pts.shape[0], 1))
        object_pts = np.hstack((object_pts, homogeneous_c))

        world_pts = h @ object_pts.T
        world_pts = world_pts / world_pts[2]

        for name, coordinates in zip(names, world_pts.T):
            point = Point(coordinates[0], coordinates[1])
            out[name] = point
        return out

    def plot(self, img, ax, marker=None, color='coral', alpha=0.4, size=100):
        """ scatter plot """
        ax.imshow(img)
        pts = self.get_numpy_arr()
        ax.scatter(pts[:, 0], pts[:, 1], alpha=alpha, c=color, marker=marker, s=size)
        return ax
    
    def plot_fov(self, img, ax, color='coral'):
        ax.imshow(img)
        pts = self.get_numpy_arr()
        ax.scatter(pts[:, 0], pts[:, 1], alpha=1, c=color, marker='x')
        ax.fill(pts[:, 0], pts[:, 1], alpha=0.5, c=color)
        return ax

    @staticmethod 
    def set_reference_pts(pv_img, tv_img, out_dir='out'):
        pv_ref_pts = ReferencePoints()
        tv_ref_pts = ReferencePoints()
        fig1, ax1, pv_ref_pts = ReferencePoints._get_reference_pt_fig(pv_ref_pts)
        fig2, ax2, tv_ref_pts = ReferencePoints._get_reference_pt_fig(tv_ref_pts)
        ax1.imshow(pv_img)
        ax2.imshow(tv_img)
        plt.show()

        print('pv reference points: ', pv_ref_pts)
        print('tv reference points: ', tv_ref_pts)

        pv_ref_pts.save(f'{out_dir}/pv_reference_pts.txt') # maybe TODO  fname
        tv_ref_pts.save(f'{out_dir}/tv_reference_pts.txt')

    @staticmethod 
    def set_fov_pts(pv_img, fname):
        fov_pts = ReferencePoints()
        fig1, ax1, fov_pts = ReferencePoints._get_reference_pt_fig(fov_pts)
        ax1.imshow(pv_img)
        plt.show()

        print('pv reference points: ', fov_pts)
        fov_pts.save(fname)


    @staticmethod
    def _get_reference_pt_fig(ref_pts):
        def on_press(event):
            sys.stdout.flush()
            if event.key == 'x':
                ax.plot(event.xdata, event.ydata, marker='x', markersize=12)
                ref_pts[len(ref_pts)+1] = Point(event.xdata, event.ydata)
                ax.text(event.xdata, event.ydata, len(ref_pts))
            if event.key == 'r':
                if len(ref_pts) > 0 and len(ax.lines) > 0:
                    ax.lines[-1].remove()
                    del_el = ref_pts.pop(len(ref_pts))
            if event.key == 'enter':
                plt.close()
            fig.canvas.draw()
        fig, ax = plt.subplots()
        fig.canvas.mpl_connect('key_press_event', on_press)
        return fig, ax, ref_pts

    @staticmethod
    def calc_homography_matrix(
            ref_pts_pv,
            ref_pts_tv
        ) -> np.ndarray:
        h, _ = cv2.findHomography(
            ref_pts_pv.get_numpy_arr(),
            ref_pts_tv.get_numpy_arr()
        )
        return h
    
    def get_rotation_and_translation_vector(
        ref_pts_tv, 
        ref_pts_pv, 
        K: np.ndarray, 
        dist: np.ndarray
    ) -> None:
        object_points = ref_pts_tv.get_numpy_arr()
        image_points = ref_pts_pv.get_numpy_arr()
        assert len(object_points) == len(image_points) and \
            not len(object_points) == 0, \
            "points not same length or 0 length."
        object_points = np.pad(
            object_points, 
            ((0, 0), (0, 1)), 
            mode = 'constant', 
            constant_values = 0
        )
        retval, rvec, tvec = cv2.solvePnP(
            object_points, 
            image_points, 
            K,
            dist
        )
        return rvec, tvec
    
    @staticmethod
    def load_K(filename: str):
        with open(filename, 'r') as file:
            data = json.load(file)
        K = data['mtx']
        dist = data['dist']
        return np.array(K), np.array(dist)
    
    @staticmethod
    def get_rotation_matrix(rvec):
        return cv2.Rodrigues(rvec)[0]

    @staticmethod
    def get_camera_position(rotM, tvec):
        return np.matmul(-rotM.T, tvec)

    
    def calc_fov(self, h):
        return self.transform(h)