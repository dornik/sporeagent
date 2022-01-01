import math
from typing import Dict
import numpy as np
from scipy.stats import special_ortho_group
from scipy.spatial.transform.rotation import Rotation
import open3d as o3d

# Adapted from RPM-Net (Yew et al., 2020): https://github.com/yewzijian/RPMNet


def uniform_2_sphere(num: int = None):
    """Uniform sampling on a 2-sphere

    Source: https://gist.github.com/andrewbolster/10274979

    Args:
        num: Number of vectors to sample (or None if single)

    Returns:
        Random Vector (np.ndarray) of size (num, 3) with norm 1.
        If num is None returned value will have size (3,)

    """
    if num is not None:
        phi = np.random.uniform(0.0, 2 * np.pi, num)
        cos_theta = np.random.uniform(-1.0, 1.0, num)
    else:
        phi = np.random.uniform(0.0, 2 * np.pi)
        cos_theta = np.random.uniform(-1.0, 1.0)

    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack((x, y, z), axis=-1)


class Resampler:
    def __init__(self, num: int):
        """Resamples a point cloud containing N points to one containing M

        Guaranteed to have no repeated points if M <= N.
        Otherwise, it is guaranteed that all points appear at least once.

        Args:
            num (int): Number of points to resample to, i.e. M

        """
        self.num = num

    def __call__(self, sample):

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if 'points' in sample:
            sample['points'] = self._resample(sample['points'], self.num)
        else:
            if 'crop_proportion' not in sample:
                src_size, ref_size = self.num, self.num
            elif len(sample['crop_proportion']) == 1:
                src_size = math.ceil(sample['crop_proportion'][0] * self.num)
                ref_size = self.num
            elif len(sample['crop_proportion']) == 2:
                src_size = math.ceil(sample['crop_proportion'][0] * self.num)
                ref_size = math.ceil(sample['crop_proportion'][1] * self.num)
            else:
                raise ValueError('Crop proportion must have 1 or 2 elements')

            sample['points_src'] = self._resample(sample['points_src'], src_size)
            sample['points_ref'] = self._resample(sample['points_ref'], ref_size)

        return sample

    @staticmethod
    def _resample(points, k):
        """Resamples the points such that there is exactly k points.

        If the input point cloud has <= k points, it is guaranteed the
        resampled point cloud contains every point in the input.
        If the input point cloud has > k points, it is guaranteed the
        resampled point cloud does not contain repeated point.
        """

        if k <= points.shape[0]:
            rand_idxs = np.random.choice(points.shape[0], k, replace=False)
            return points[rand_idxs, :]
        elif points.shape[0] == k:
            return points
        else:
            rand_idxs = np.concatenate([np.random.choice(points.shape[0], points.shape[0], replace=False),
                                        np.random.choice(points.shape[0], k - points.shape[0], replace=True)])
            return points[rand_idxs, :]


class TransformSE3:
    def __init__(self):
        """Applies a random rigid transformation to the source point cloud"""

    def apply_transform(self, p0, transform_mat):
        p1 = (transform_mat[:3, :3] @ p0[:, :3].T).T + transform_mat[:3, 3]
        if p0.shape[1] >= 6:  # Need to rotate normals too
            n1 = (transform_mat[:3, :3] @ p0[:, 3:6].T).T
            p1 = np.concatenate((p1, n1), axis=-1)
        if p0.shape[1] == 4:  # label (pose estimation task)
            p1 = np.concatenate((p1, p0[:, 3][:, None]), axis=-1)
        if p0.shape[1] > 6:  # additional channels after normals
            p1 = np.concatenate((p1, p0[:, 6:]), axis=-1)

        igt = transform_mat
        # invert to get gt
        gt = igt.copy()
        gt[:3, :3] = gt[:3, :3].T
        gt[:3, 3] = -gt[:3, :3] @ gt[:3, 3]

        return p1, gt, igt

    def __call__(self, sample):
        raise NotImplementedError("Subclasses implement transformation (random, given, etc).")


class RandomTransformSE3(TransformSE3):
    def __init__(self, rot_mag: float = 45.0, trans_mag: float = 0.5, random_mag: bool = False):
        """Applies a random rigid transformation to the source point cloud

        Args:
            rot_mag (float): Maximum rotation in degrees
            trans_mag (float): Maximum translation T. Random translation will
              be in the range [-X,X] in each axis
            random_mag (bool): If true, will randomize the maximum rotation, i.e. will bias towards small
                               perturbations
        """
        super().__init__()
        self._rot_mag = rot_mag
        self._trans_mag = trans_mag
        self._random_mag = random_mag

    def generate_transform(self, rot_mag=None, trans_mag=None):
        """Generate a random SE3 transformation (3, 4) """

        if rot_mag is None or trans_mag is None:
            if self._random_mag:
                rot_mag, trans_mag = np.random.uniform() * self._rot_mag, np.random.uniform() * self._trans_mag
            else:
                rot_mag, trans_mag = self._rot_mag, self._trans_mag

        # Generate rotation
        rand_rot = special_ortho_group.rvs(3)
        axis_angle = Rotation.as_rotvec(Rotation.from_dcm(rand_rot))
        axis_angle /= np.linalg.norm(axis_angle)
        axis_angle *= np.deg2rad(rot_mag)
        rand_rot = Rotation.from_rotvec(axis_angle).as_dcm()

        # Generate translation
        rand_trans = uniform_2_sphere()
        if self._random_mag:
            rand_trans *= np.random.uniform(high=trans_mag)
        else:
            rand_trans *= trans_mag
        rand_SE3 = np.concatenate((rand_rot, rand_trans[:, None]), axis=1).astype(np.float32)

        return rand_SE3

    def transform(self, tensor):
        transform_mat = self.generate_transform()
        return self.apply_transform(tensor, transform_mat)

    def __call__(self, sample):

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if 'points' in sample:
            sample['points'], _, _ = self.transform(sample['points'])
        else:
            transformed, transform_r_s, transform_s_r = self.transform(sample['points_src'])
            sample['transform_gt'] = transform_r_s  # Apply to source to get reference
            sample['points_src'] = transformed

            # update relative trafo between estimated points -> (from GT=est plane to GT target, now move to est target)
            target_ref2src = np.eye(4, dtype=np.float32)
            target_ref2src[:3, :] = transform_s_r
            target_ref2src = sample['normalization'] @ target_ref2src @ np.linalg.inv(sample['normalization'])
            sample['relative_plane2points'] = target_ref2src @ sample['relative_plane2points']

        return sample


class SetDeterministic:
    """Adds a deterministic flag to the sample such that subsequent transforms
    use a fixed random seed where applicable. Used for test"""

    def __call__(self, sample):
        sample['deterministic'] = True
        return sample


# Additional augmentations proposed in SporeAgent

class Normalize:
    """Normalizes source and target to be mean-centered and scales s.t. farthest point is of distance 1."""

    def __init__(self, using_target=True):
        self.using_target = using_target  # normalize wrt target

    def __call__(self, sample):
        t = sample['points_ref'][:, :3].mean(axis=0)  # center offset
        centered = sample['points_ref'][:, :3] - t
        dists = np.linalg.norm(centered, axis=1)
        s = dists.max()  # scale

        # apply to source and target
        sample['points_ref'][:, :3] = centered / s
        sample['points_src'][:, :3] = (sample['points_src'][:, :3] - t) / s

        # for test set with given estimate in unnormalized scale, adapt translation scale
        if 'transform_gt' in sample:
            sample['transform_gt'][:3, 3] /= s

        # keep track (to undo if needed)
        sample['normalization'] = np.eye(4, dtype=np.float32)
        sample['normalization'][np.diag_indices(3)] = s
        sample['normalization'][:3, 3] = t.squeeze()

        # -- plane
        # scale s.t. ref is within unit sphere (already centered at origin -> no translation)
        s_plane = np.linalg.norm(sample['plane_ref'], axis=1).max()
        sample['plane_src'][:, :3] = sample['plane_src'][:, :3] / s_plane
        sample['plane_ref'][:, :3] = sample['plane_ref'][:, :3] / s_plane
        # keep track
        sample['plane_normalization'] = np.eye(4, dtype=np.float32)
        sample['plane_normalization'][np.diag_indices(3)] = s_plane
        sample['scale_plane2points'] = s_plane / s

        # for test set with given estimate in unnormalized scale
        if 'plane_transform_gt' in sample:
            sample['plane_transform_gt'][:3, 3] /= s_plane

        return sample


class GtTransformSE3(TransformSE3):
    """Takes transformation from GT dictionary and applies it to source/target for initial alignment."""

    def __init__(self, align_plane=False):
        super().__init__()
        self.align_plane = align_plane  # try to align coordinate systems for computed plane pose

    def __call__(self, sample):
        cam2model = np.linalg.inv(sample['gt_m2c'])
        sample['points_src'], sample['est_m2c'], sample['est_c2m'] = self.apply_transform(sample['points_src'], cam2model)

        # -- get relative transformation between estimated poses -- assuming sample['plane_m2c'] estimate is the GT
        # plane(est)->target(est)  (target(est)=target(gt) at the moment - error added later)
        sample['relative_plane2points'] = cam2model @ sample['plane_m2c']

        if self.align_plane:  # center plane(est) for LM; rotate by 90deg around z to align coord systems
            offset = np.eye(4, dtype=np.float32)
            offset[:2, 3] = self.apply_transform(sample['points_src'],
                                                 np.linalg.inv(sample['relative_plane2points']))[0][:, :2].mean(axis=0).squeeze()
            offset[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
            sample['plane_m2c'] = sample['plane_m2c'] @ offset
            sample['relative_plane2points'] = sample['relative_plane2points'] @ offset

        # -- transform observed plane from camera space to plane space
        sample['plane_src'], _, _ = self.apply_transform(sample['plane_src'], np.linalg.inv(sample['plane_m2c']))

        return sample


class EstTransformSE3(TransformSE3):
    """Takes transformation from estimate and applies it to source/target for initial alignment."""

    def __init__(self, align_plane=False):
        super().__init__()
        self.align_plane = align_plane

    def __call__(self, sample):
        # this is our actual estimate -- it's residual from the best estimate is the initial pose error
        est = sample['est_m2c']
        inv_est = np.linalg.inv(est)  # model2cam -> cam2model -- this is then applied to src points (cam space)
        sample['points_src'], transform_r_s, transform_s_r = self.apply_transform(sample['points_src'], inv_est[:3, :])

        cam2model = np.linalg.inv(sample['gt_m2c'])
        sample['transform_gt'] = (cam2model @ est)[:3, :]  # apply to source to get reference

        # plane is estimated in camera frame and then transformed to estimated model frame via estimate
        if 'plane_m2c' in sample:
            # cam2model-est @ plane-est2cam = plane(est)->target(est)
            sample['relative_plane2points'] = inv_est @ sample['plane_m2c']

            # target(est) in plane(est) --> center plane(est) for LM; rotate by 90deg around z to align coord systems
            if self.align_plane:
                offset = np.eye(4, dtype=np.float32)
                offset[:2, 3] = self.apply_transform(sample['points_src'],
                                                     np.linalg.inv(sample['relative_plane2points']))[0][:, :2].mean(axis=0).squeeze()
                offset[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
                sample['plane_m2c'] = sample['plane_m2c'] @ offset
                sample['relative_plane2points'] = sample['relative_plane2points'] @ offset

            # observed scene to est plane coords
            sample['plane_src'], _, _ = self.apply_transform(sample['plane_src'], np.linalg.inv(sample['plane_m2c']))

            return sample


class SegmentResampler(Resampler):
    """Simulates imprecise segmentation by sampling nearest-neighbors to a random seed within the segmentation mask."""

    def __init__(self, num, p_fg=1.0, patch=True):
        super().__init__(num)
        self.p_fg = p_fg
        self.patch = patch  # sample continuous patch if true, else sample randomly in cloud

    def _patch_sample(self, fg_points, fg_size, bg_points, bg_size, K):
        # assuming points_src to be in camera space, we project to image space for kNN search
        def project(points):
            Xs, Ys, Zs = points[:, 0], points[:, 1], points[:, 2]
            xs = np.divide(Xs * K[0, 0], Zs) + K[0, 2]
            ys = np.divide(Ys * K[1, 1], Zs) + K[1, 2]
            return np.hstack([xs[:, None], ys[:, None]])

        fg_2d = project(fg_points)

        # pick a random point on the object as center
        center_idx = np.random.randint(0, fg_points.shape[0])
        center_2d = fg_2d[center_idx]

        # find [fg_size] nearest points
        fg_centered = fg_2d - center_2d
        fg_distance = np.linalg.norm(fg_centered, axis=1)
        fg_neighbors = np.argsort(fg_distance)[:fg_size]
        fg = fg_points[fg_neighbors]

        if bg_size > 0:
            bg_2d = project(bg_points)
            # find [bg_size] nearest points
            bg_centered = bg_2d - center_2d
            bg_distance = np.linalg.norm(bg_centered, axis=1)
            bg_neighbors = np.argsort(bg_distance)[:bg_size]
            bg = bg_points[bg_neighbors]
        else:
            bg = []
            bg_neighbors = []

        return fg, bg, fg_neighbors, bg_neighbors

    def __call__(self, sample: Dict):
        if sample['obj_id'] == 0:  # padding
            sample['points_src'] = self._resample(sample['points_src'], self.num)
        else:  # [p_fg]% from object, [1-p_fg]% from background
            fg = sample['points_src'][:, -1] > 0
            bg = sample['points_src'][:, -1] == 0
            if isinstance(self.p_fg, list):
                p_fg = np.random.uniform(*self.p_fg)
            else:
                p_fg = self.p_fg
            fg_size = math.ceil(p_fg * self.num)
            bg_size = math.floor((1 - p_fg) * self.num)
            assert fg_size + bg_size == self.num
            if self.patch:
                fg, bg, _, _ = self._patch_sample(sample['points_src'][fg], fg_size,
                                                  sample['points_src'][bg], bg_size,
                                                  sample['cam']['cam_K'])
            else:
                fg = self._resample(sample['points_src'][fg], fg_size)
                bg = self._resample(sample['points_src'][bg], bg_size)

            if fg_size > 0 and bg_size > 0:
                sample['points_src'] = np.vstack([fg, bg])
            elif fg_size > 0:
                sample['points_src'] = fg
            else:
                raise ValueError("only background pixels sampled")

        # num from model
        sample['points_ref'] = self._resample(sample['points_ref'], self.num)

        # --- plane/scene
        sample['plane_src'] = self._resample(sample['plane_src'][sample['plane_src'][:, -1] == 1], self.num)  # no obj
        sample['plane_ref'] = self._resample(sample['plane_ref'], self.num)

        return sample


class RandomTransformSE3_plane(RandomTransformSE3):
    """Random transform for plane"""

    def generate_transform_plane(self):
        if self._random_mag:
            rot_mag, trans_mag = np.random.uniform() * self._rot_mag, np.random.uniform() * self._trans_mag
        else:
            rot_mag, trans_mag = self._rot_mag, self._trans_mag

        transform_err_plane = np.eye(4, dtype=np.float32)
        transform_err_plane[:3, :] = self.generate_transform(rot_mag, trans_mag)
        return transform_err_plane

    def __call__(self, sample):
        if 'plane_err' not in sample:
            transform_err_plane = self.generate_transform_plane()
        else:  # used to set the same plane (error) for all objects/samples in a single frame
            transform_err_plane = sample['plane_err']
        # assumes error given in meters -> to mm, normalize (so we can define in mm and apply in normalized)
        transform_err_plane[:3, 3] *= 1000 / sample['plane_normalization'][0, 0]

        if 'plane_src' in sample:
            sample['plane_src'], _, _ = self.apply_transform(sample['plane_src'], transform_err_plane)
            sample['plane_transform_gt'] = np.linalg.inv(transform_err_plane)

            # update relative trafo between estimated points -> from GT plane, move to est plane (then to target)
            plane_gt = sample['plane_normalization'] @ sample['plane_transform_gt'] @ np.linalg.inv(sample['plane_normalization'])
            sample['relative_plane2points'] = sample['relative_plane2points'] @ plane_gt

        return sample


class ComputeNormals:
    """Estimate normals from point cloud using Open3D."""

    def __call__(self, sample: Dict):
        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample['points_src'][:, :3]))
        src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
        src.orient_normals_to_align_with_direction([0, 0, -1])
        sample['points_src'] = np.hstack([sample['points_src'][:, :3], np.asarray(src.normals).astype(np.float32),
                                          sample['points_src'][:, -1][:, None]])  # mask
        if 'plane_src' in sample:
            plane_src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample['plane_src'][:, :3]))
            plane_src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=50, max_nn=30))
            plane_src.orient_normals_to_align_with_direction([0, 0, -1])
            sample['plane_src'] = np.hstack([sample['plane_src'][:, :3], np.asarray(plane_src.normals).astype(np.float32),
                                             sample['plane_src'][:, -1][:, None]])  # mask
        return sample
