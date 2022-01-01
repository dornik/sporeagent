import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import os
import pickle
import trimesh
import sys
import skimage.io as skio
from tqdm import tqdm
import open3d as o3d
import cv2 as cv
from scipy.spatial.transform.rotation import Rotation
import glob

import config as cfg
import dataset.augmentation as Transforms

sys.path.append(cfg.BOP_PATH)
sys.path.append(os.path.join(cfg.BOP_PATH, "bop_toolkit_lib"))
import bop_toolkit_lib.inout as bop_inout
import bop_toolkit_lib.dataset_params as bop_dataset_params


class DatasetLinemod(Dataset):

    def __init__(self, split, dataset_path=cfg.LM_PATH):
        self.split_name = split
        subsample = 16 if split == "eval" else 0  # use every 16th test sample for evaluation during training
        split = "test" if split == "eval" else split
        self.split = split
        self.dataset_path = dataset_path
        self.samples, self.models, self.symmetries = self.get_samples(split, subsample)
        self.transforms = self.get_transforms(split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # compose sample
        model = self.models[item['obj_id']]
        gt = np.eye(4, dtype=np.float32)
        gt[:3, :3] = item['gt']['cam_R_m2c']
        gt[:3, 3] = item['gt']['cam_t_m2c'].squeeze()

        sample = {
            'idx': idx,
            'points_src': item['pcd'],
            'points_ref': model,
            'scene': item['scene'],
            'frame': item['frame'],
            'cam': item['cam'],
            'obj_id': item['gt']['obj_id'],
            'gt_m2c': gt,
            'plane_src': item['plane_pcd'],
            'plane_ref': self.models[0]
        }
        if 'est' in item:  # initial estimate only given for test split (using PoseCNN)
            sample['est_m2c'] = item['est']
        if self.symmetries is not None:
            sample['symmetries'] = self.symmetries[item['obj_id']]['symmetries']  # padded to max number of syms
            sample['num_symmetries'] = self.symmetries[item['obj_id']][
                'num_symmetries']  # num of valid syms (rest Id)
        if self.split != 'test' and 'cam_R_w2c' in item['cam']:  # use annotated supporting plane for training
            sample['plane_m2c'] = np.eye(4, dtype=np.float32)
            sample['plane_m2c'][:3, :3] = sample['cam']['cam_R_w2c']
            sample['plane_m2c'][:3, 3] = sample['cam']['cam_t_w2c'].squeeze()
        elif 'plane' in item:  # pre-computed supporting plane (RANSAC) when not available and for testing
            sample['plane_m2c'] = item['plane']

        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def get_transforms(self, split):
        # error/noise magnitudes per dataset
        if self.dataset_path == cfg.YCBV_PATH:
            rot_mag, trans_mag = 75.0, 0.75
            p_fg = [0.8, 1.0]
        else:
            rot_mag, trans_mag = 90.0, 1.0
            p_fg = [0.5, 1.0]

        # prepare augmentations
        if split == "train":
            transforms = [
                # resample segmentation (with [p_fg]% from object)
                Transforms.SegmentResampler(1024, p_fg=p_fg),
                # align source and target using GT -- easier to define error this way
                Transforms.GtTransformSE3(align_plane=self.dataset_path == cfg.LM_PATH),
                # normalize source and target (mean centered, max dist 1.0)
                Transforms.Normalize(),
                # apply an initial pose error
                Transforms.RandomTransformSE3(rot_mag=rot_mag, trans_mag=trans_mag, random_mag=True),
            ]
            if cfg.USE_CONTACT:
                transforms.append(Transforms.RandomTransformSE3_plane(rot_mag=5.0, trans_mag=0.02, random_mag=True))
            # note: re-computing them from resampled pcds increases variance
            if cfg.USE_NORMALS and not cfg.PRECOMPUTE_NORMALS:
                transforms.insert(1, Transforms.ComputeNormals())
        elif split == "val":
            transforms = [
                Transforms.SetDeterministic(),
                Transforms.SegmentResampler(1024, p_fg=p_fg),
                Transforms.GtTransformSE3(align_plane=self.dataset_path == cfg.LM_PATH),
                Transforms.Normalize(),
                Transforms.RandomTransformSE3(rot_mag=rot_mag, trans_mag=trans_mag, random_mag=True),
            ]
            if cfg.USE_CONTACT:
                transforms.append(Transforms.RandomTransformSE3_plane(rot_mag=5.0, trans_mag=0.02, random_mag=True))
            if cfg.USE_NORMALS and not cfg.PRECOMPUTE_NORMALS:
                transforms.insert(2, Transforms.ComputeNormals())
        else:  # start from posecnn
            transforms = [
                Transforms.SetDeterministic(),
                # randomly resample inside segmentation mask (estimated by PoseCNN)
                Transforms.SegmentResampler(1024, p_fg=1.0, patch=False),  # note: fg is predicted mask
                Transforms.EstTransformSE3(align_plane=self.dataset_path == cfg.LM_PATH),
                Transforms.Normalize()
            ]
            if cfg.USE_NORMALS and not cfg.PRECOMPUTE_NORMALS:
                transforms.insert(2, Transforms.ComputeNormals())
        return torchvision.transforms.Compose(transforms)

    def get_samples(self, split, subsample=0):
        # ============= GET MODELS ============
        model_type = "eval"
        model_params = bop_dataset_params.get_model_params('/'.join(self.dataset_path.split('/')[:-1]),
                                                           self.dataset_path.split('/')[-1], model_type)
        mesh_ids = model_params['obj_ids']

        # plane as 0 object: xy plane, z up
        plane_mm = 1000/np.sqrt(2)  # size in mm
        plane_samples = 4096
        plane = np.hstack([v.reshape(-1, 1)
                           for v in np.meshgrid(*[np.linspace(-plane_mm/2, plane_mm/2, int(np.sqrt(plane_samples)))]*2)]
                          + [np.zeros((plane_samples, 3)), np.ones((plane_samples, 1))]).astype(np.float32)
        models = {0: plane}
        for mesh_id in mesh_ids:
            mesh = trimesh.load(os.path.join(self.dataset_path, f"models_{model_type}/obj_{mesh_id:06d}.ply"))
            pcd, face_indices = trimesh.sample.sample_surface_even(mesh, 4096)
            if pcd.shape[0] < 4096:  # pad by additional samples if less than 4096 were returned
                additional_samples = np.random.choice(np.arange(pcd.shape[0]), size=4096 - pcd.shape[0], replace=True)
                pcd = np.vstack([pcd, pcd[additional_samples]])
                face_indices = np.hstack([face_indices, face_indices[additional_samples]])
            models[mesh_id] = np.hstack([pcd, mesh.face_normals[face_indices]]).astype(np.float32)

        # ============= GET DATASET SAMPLES ============
        samples_path = f"sporeagent/{split}_posecnn_plane.pkl" if split == "test"\
            else f"sporeagent/{split}_plane.pkl"
        with open(os.path.join(self.dataset_path, samples_path), 'rb') as file:
            samples = pickle.load(file)
        if subsample > 0:  # used for evaluation during training
            samples = samples[::subsample]

        if cfg.PRECOMPUTE_NORMALS:
            print(f"precomputing normals for {len(samples)} samples...")
            for sample in tqdm(samples):
                src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample['pcd'][:, :3]))
                src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
                src.orient_normals_to_align_with_direction([0, 0, -1])
                sample['pcd'] = np.hstack(
                    [sample['pcd'][:, :3], np.asarray(src.normals).astype(np.float32),
                     sample['pcd'][:, -1][:, None]])  # mask
                if 'plane_pcd' in sample:
                    plane_src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample['plane_pcd'][:, :3]))
                    plane_src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=50, max_nn=30))
                    plane_src.orient_normals_to_align_with_direction([0, 0, -1])
                    sample['plane_pcd'] = np.hstack(
                        [sample['plane_pcd'][:, :3], np.asarray(plane_src.normals).astype(np.float32),
                         sample['plane_pcd'][:, -1][:, None]])  # mask

        if cfg.USE_SYMMETRY:  # add symmetry information from models_info.json
            meta_sym = bop_inout.load_json(model_params['models_info_path'], True)
            obj_symmetries = {0: []}
            for obj_id in models.keys():
                if obj_id == 0:
                    continue
                elif self.dataset_path == cfg.LM_PATH and obj_id in [3, 7]:
                    continue
                # always add identity as one of the correct poses
                obj_symmetries[obj_id] = [np.eye(4, dtype=np.float32)]
                # note: assumes that we have either one or the other -- no combination done here
                if 'symmetries_continuous' in meta_sym[obj_id]:
                    # sample rotations about the given axis
                    axis = 'xyz'[meta_sym[obj_id]['symmetries_continuous'][0]['axis'].index(1)]
                    sym = np.eye(4, dtype=np.float32)
                    sym[:3, 3] = np.asarray(meta_sym[obj_id]['symmetries_continuous'][0]['offset'])
                    axis_symmetries = []
                    for angle in range(cfg.SYMMETRY_AXIS_DELTA, 360, cfg.SYMMETRY_AXIS_DELTA):
                        sym_step = sym.copy()
                        sym_step[:3, :3] = Rotation.from_euler(axis, angle, degrees=True).as_matrix()
                        axis_symmetries.append(sym_step)
                    obj_symmetries[obj_id] += axis_symmetries
                elif 'symmetries_discrete' in meta_sym[obj_id]:
                    obj_symmetries[obj_id] += [np.asarray(sym).reshape(4, 4).astype(np.float32)
                                           for sym in meta_sym[obj_id]['symmetries_discrete']]
            # pad to max number of symmetries (for default pytorch collate_fn) and get symmetry count (for retrieval)
            max_num_symmetries = max([len(syms) for syms in obj_symmetries.values()])
            symmetries = {}
            for obj_id, syms in obj_symmetries.items():
                num_symmetries = len(syms)
                symmetries[obj_id] = {
                    'symmetries': np.stack(syms
                                           + [np.eye(4, dtype=np.float32)] * (max_num_symmetries - num_symmetries)),
                    'num_symmetries': num_symmetries
                }
        else:
            symmetries = None

        return samples, models, symmetries

    # for visualization
    def get_rgb(self, scene_id, im_id):
        test_path = "test"
        scene_path = os.path.join(self.dataset_path, f"{test_path}/{scene_id:06d}")
        file_path = os.path.join(scene_path, f"rgb/{im_id:06d}.png")
        if os.path.exists(file_path):
            return bop_inout.load_im(file_path)[..., :3]/255
        else:
            print(f"missing file: {file_path}")
            return np.zeros((480, 640, 3), dtype=np.float32)

    def get_depth(self, scene_id, im_id, depth_scale=1.0):
        test_path = "test"
        scene_path = os.path.join(self.dataset_path, f"{test_path}/{scene_id:06d}")
        file_path = os.path.join(scene_path, f"depth/{im_id:06d}.png")
        if os.path.exists(file_path):
            return bop_inout.load_depth(file_path) * depth_scale
        else:
            print(f"missing file: {file_path}")
            return np.zeros((480, 640), dtype=np.float32)

    def get_normal(self, scene_id, im_id, depth_scale=1.0):
        test_path = "test"
        scene_path = os.path.join(self.dataset_path, f"{test_path}/{scene_id:06d}")
        file_path = os.path.join(scene_path, f"normal/{im_id:06d}.tiff")
        if os.path.exists(file_path):
            return skio.imread(file_path)
        else:
            print(f"missing file: {file_path} -- trying to compute from depth")
            basepath = "/".join(file_path.split("/")[:-1])
            if not os.path.exists(basepath):
                os.mkdir(basepath)

            D = self.get_depth(scene_id, im_id, depth_scale)
            D_px = D.copy()
            # inpaint missing depth values
            D_px = cv.inpaint(D_px.astype(np.float32), np.uint8(D_px == 0), 3, cv.INPAINT_NS)
            # blur
            blur_size = (9, 9) if self.dataset_path == cfg.YCBV_PATH else (3, 3)
            D_px = cv.GaussianBlur(D_px, blur_size, sigmaX=10.0)
            # get derivatives
            kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            dzdx = cv.filter2D(D_px, -1, kernelx)
            dzdy = cv.filter2D(D_px, -1, kernely)
            # gradient ~ normal
            normal = np.dstack((dzdy, dzdx, D_px != 0.0))  # only where we have a depth value
            n = np.linalg.norm(normal, axis=2)
            n = np.dstack((n, n, n))
            normal = np.divide(normal, n, where=(n != 0))
            # remove invalid values
            normal[n == 0] = 0.0
            normal[D == 0] = 0.0
            # save normals for next use
            skio.imsave(file_path, normal)
            return normal

    def get_seg(self, scene_id, im_id, gt_id=-1):
        scene_path = os.path.join(self.dataset_path, f"test/{scene_id:06d}")
        if gt_id >= 0:
            file_path = os.path.join(scene_path, f"mask_visib/{im_id:06d}_{gt_id:06d}.png")
            if os.path.exists(file_path):
                return bop_inout.load_im(file_path)
            else:
                print(f"missing file: {file_path}")
                return np.zeros((480, 640), dtype=np.uint8)
        else:
            file_paths = sorted(glob.glob(os.path.join(scene_path, f"mask_visib/{im_id:06d}_*.png")))
            meta = bop_inout.load_json(os.path.join(self.dataset_path, f"test/{scene_id:06d}/scene_gt.json"))[str(im_id)]
            obj_ids = [info['obj_id'] for info in meta]
            masks = [bop_inout.load_im(file_path) for file_path in file_paths]
            labels = np.zeros((480, 640), dtype=np.uint8)
            for (mask, obj_id) in zip(masks, obj_ids):
                labels[mask > 0] = obj_id
            return labels


class DatasetYcbVideo(DatasetLinemod):

    def __init__(self, split, dataset_path=cfg.YCBV_PATH):
        super().__init__(split, dataset_path)

    def __getitem__(self, idx):
        frame_samples = self.samples[idx]
        if self.transforms:
            # same plane for all objects in the frame --> same error applied to GT for train and val
            if cfg.USE_CONTACT and self.split != 'test':
                # assert isinstance(self.transforms.transforms[-1], Transforms.RandomTransformSE3_plane)
                plane_err = self.transforms.transforms[-1].generate_transform_plane()
                for sample in frame_samples:
                    sample['plane_err'] = plane_err

            frame_samples = [self.transforms(obj_sample.copy()) for obj_sample in frame_samples]
        return frame_samples

    def get_samples(self, split, subsample=0):
        # ============= GET MODELS ============
        model_type = "eval_canonical" if cfg.USE_CANONICAL else "eval"
        model_params = bop_dataset_params.get_model_params('/'.join(self.dataset_path.split('/')[:-1]),
                                                           self.dataset_path.split('/')[-1], model_type)
        mesh_ids = model_params['obj_ids']

        # plane as 0 object: xy plane, z up
        plane_mm = 1000/np.sqrt(2)  # scale in mm
        plane_samples = 4096
        plane = np.hstack([v.reshape(-1, 1)
                           for v in
                           np.meshgrid(*[np.linspace(-plane_mm / 2, plane_mm / 2, int(np.sqrt(plane_samples)))] * 2)]
                          + [np.zeros((plane_samples, 3)), np.ones((plane_samples, 1))]).astype(np.float32)
        models = {0: plane}
        for mesh_id in mesh_ids:
            mesh = trimesh.load(os.path.join(self.dataset_path, f"models_{model_type}/obj_{mesh_id:06d}.ply"))
            pcd, face_indices = trimesh.sample.sample_surface_even(mesh, 4096)
            if pcd.shape[0] < 4096:  # get additional samples if less were returned
                additional_samples = np.random.choice(np.arange(pcd.shape[0]), size=4096 - pcd.shape[0], replace=True)
                pcd = np.vstack([pcd, pcd[additional_samples]])
                face_indices = np.hstack([face_indices, face_indices[additional_samples]])
            models[mesh_id] = np.hstack([pcd, mesh.face_normals[face_indices]]).astype(np.float32)

        # ============= ADD SYMMETRY INFORMATION ============
        if cfg.USE_SYMMETRY:  # add symmetry information from models_info.json
            from scipy.spatial.transform.rotation import Rotation
            info_path = model_params['models_info_path']
            info_path = info_path.replace('models_eval', 'models_eval_canonical') if not cfg.USE_CANONICAL else info_path
            meta_sym = bop_inout.load_json(info_path, True)
            obj_symmetries = {0: []}
            for obj_id in models.keys():
                if obj_id == 0:
                    continue
                elif self.dataset_path == cfg.LM_PATH and obj_id in [3, 7]:
                    continue
                # always add identity as one of the correct poses
                obj_symmetries[obj_id] = [np.eye(4, dtype=np.float32)]
                # note: assumes that we have either one or the other -- no combination done here
                if 'symmetries_continuous' in meta_sym[obj_id]:
                    # sample rotations about the given axis
                    axis = 'xyz'[meta_sym[obj_id]['symmetries_continuous'][0]['axis'].index(1)]
                    sym = np.eye(4, dtype=np.float32)
                    sym[:3, 3] = np.asarray(meta_sym[obj_id]['symmetries_continuous'][0]['offset'])
                    axis_symmetries = []
                    for angle in range(cfg.SYMMETRY_AXIS_DELTA, 360, cfg.SYMMETRY_AXIS_DELTA):
                        sym_step = sym.copy()
                        sym_step[:3, :3] = Rotation.from_euler(axis, angle, degrees=True).as_matrix()
                        axis_symmetries.append(sym_step)
                    obj_symmetries[obj_id] += axis_symmetries
                    # cylinder -- both
                    if 'symmetries_discrete' in meta_sym[obj_id]:
                        syms = [np.asarray(sym).reshape(4, 4).astype(np.float32)
                                               for sym in meta_sym[obj_id]['symmetries_discrete']]
                        assert len(syms) == 1
                        up_down = [syms[0] @ axis_sym for axis_sym in axis_symmetries]
                        obj_symmetries[obj_id] += up_down

                elif 'symmetries_discrete' in meta_sym[obj_id]:
                    obj_symmetries[obj_id] += [np.asarray(sym).reshape(4, 4).astype(np.float32)
                                               for sym in meta_sym[obj_id]['symmetries_discrete']]
            # pad to max number of symmetries (for default pytorch collate_fn) and get symmetry count (for retrieval)
            max_num_symmetries = max([len(syms) for syms in obj_symmetries.values()])
            symmetries = {}
            for obj_id, syms in obj_symmetries.items():
                num_symmetries = len(syms)
                symmetries[obj_id] = {
                    'symmetries': np.stack(syms
                                           + [np.eye(4, dtype=np.float32)] * (max_num_symmetries - num_symmetries)),
                    'num_symmetries': num_symmetries
                }
        else:
            symmetries = None

        # ============= PREP CANONICAL INFO ==========
        # get GT to canonical model space trafos
        code_path = os.path.dirname(os.path.abspath(__file__))
        meta_canon = bop_inout.load_json(os.path.join(code_path, "bop_models_meta.json"), True)
        canonical_offsets = {}
        for obj_id in models.keys():
            if obj_id == 0:
                continue
            bop_to_canonical = np.eye(4, dtype=np.float32)
            bop_to_canonical[:3, :3] = np.asarray(meta_canon[obj_id]['R_to_canonical']).reshape(3, 3)
            bop_to_canonical[:3, 3] = np.asarray(meta_canon[obj_id]['t_to_canonical']).squeeze()
            canonical_offsets[obj_id] = np.linalg.inv(bop_to_canonical)
        if not cfg.USE_CANONICAL and cfg.USE_SYMMETRY:  # adapt symmetries accordingly
            bop_symmetries = {}
            for obj_id, syms in symmetries.items():
                if obj_id == 0:
                    continue
                to_canon = canonical_offsets[obj_id]

                bop_symmetries[obj_id] = {'num_symmetries': symmetries[obj_id]['num_symmetries']}
                canon_symmetries = []
                for sym in symmetries[obj_id]['symmetries']:
                    sym = to_canon @ sym @ np.linalg.inv(to_canon)
                    canon_symmetries.append(sym)
                bop_symmetries[obj_id]['symmetries'] = np.stack(canon_symmetries)
            symmetries = bop_symmetries

        # ============= GET DATASET SAMPLES ============
        samples_path = f"sporeagent/{split}_posecnn_plane.pkl" if split == "test" \
            else f"sporeagent/{split}_plane.pkl"
        with open(os.path.join(self.dataset_path, samples_path), 'rb') as file:
            samples = pickle.load(file)
        if subsample > 0:  # used for evaluation during training
            samples = samples[::subsample]
        # we precompute samples for 1/20th of the training set
        if self.split_name != "test":
            samples = samples[::5]  # further subsample to 1/100th during training

        print(f"preparing {len(samples)} {self.split_name} samples...")
        pad_size = max([len(sample['obj_ids']) for sample in samples])
        per_frame_samples = []
        for sample in tqdm(samples):
            # prepare plane: optionally use provided plane pose for training (where given) and precompute normals
            if self.split != 'test' and 'cam_R_w2c' in sample['cam']:  # use annotated supporting plane for training
                sample['plane'] = np.eye(4, dtype=np.float32)
                sample['plane'][:3, :3] = sample['cam']['cam_R_w2c']
                sample['plane'][:3, 3] = sample['cam']['cam_t_w2c'].squeeze()
            # else: pre-computed supporting plane (RANSAC) when not available and for testing
            if cfg.PRECOMPUTE_NORMALS:  # for observed plane
                plane_src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample['plane_pcd'][:, :3]))
                plane_src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=50, max_nn=30))
                plane_src.orient_normals_to_align_with_direction([0, 0, -1])
                sample['plane_pcd'] = np.hstack(
                    [sample['plane_pcd'][:, :3], np.asarray(plane_src.normals).astype(np.float32),
                     sample['plane_pcd'][:, -1][:, None]])  # mask
            sample['plane_pcd'][:, -1] = sample['plane_pcd'][:, -1] == 0

            # unpack objects per frame
            num_objs = len(sample['obj_ids'])
            frame_samples = []
            for oi, obj_id in enumerate(sample['obj_ids']):
                m2c = np.eye(4, dtype=np.float32)
                m2c[:3, :3] = sample['gts'][oi]['cam_R_m2c']
                m2c[:3, 3] = sample['gts'][oi]['cam_t_m2c'].squeeze()

                if cfg.USE_CANONICAL:  # adapt GT and est to canonical model space
                    m2c = m2c @ canonical_offsets[obj_id]
                    sample['ests'][oi] = sample['ests'][oi] @ canonical_offsets[obj_id]

                if cfg.PRECOMPUTE_NORMALS:  # for observed objects
                    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample['pcds'][oi][:, :3]))
                    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
                    src.orient_normals_to_align_with_direction([0, 0, -1])
                    sample['pcds'][oi] = np.hstack(
                        [sample['pcds'][oi][:, :3], np.asarray(src.normals).astype(np.float32),
                         sample['pcds'][oi][:, -1][:, None]])  # mask

                obj_sample = {
                    'idx': len(per_frame_samples) * pad_size + oi,
                    'scene': sample['scene'], 'frame': sample['frame'], 'obj_id': sample['gts'][oi]['obj_id'],
                    'cam': sample['cam'], 'gt_m2c': m2c, 'est_m2c': sample['ests'][oi],
                    'plane_src': sample['plane_pcd'], 'plane_ref': models[0], 'plane_m2c': sample['plane'],
                    'num_frame_objects': num_objs,
                    'other_obj_ids': sample['obj_ids'][:oi] + sample['obj_ids'][oi + 1:] + [0] * (pad_size - num_objs),
                    'points_src': sample['pcds'][oi], 'points_ref': models[obj_id]
                }

                if symmetries is not None:
                    obj_sample['symmetries'] = symmetries[obj_id]['symmetries']
                    obj_sample['num_symmetries'] = symmetries[obj_id]['num_symmetries']

                frame_samples.append(obj_sample)
            per_frame_samples.append(frame_samples)
        return per_frame_samples, models, symmetries


def collate_data(data):
    if isinstance(data[0], list):  # custom for per-frame samples (i.e., for YCBV)
        batch = []
        for frame in data:
            batch += frame
        batch = torch.utils.data.dataloader.default_collate(batch)
        # fix 'other_obj_ids'
        batch['other_obj_ids'] = torch.cat([other_obj_ids[:, None] for other_obj_ids in batch['other_obj_ids']], dim=1)
    else:
        batch = torch.utils.data.dataloader.default_collate(data)
    return batch
