import torch
import os
import argparse
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)).replace("/registration", ""))
import dataset.dataset
import config as cfg
from environment import environment as env
from environment import transformations as tra
from registration.model import Agent
import registration.model as util_model
sys.path.append(cfg.BOP_PATH)
sys.path.append(os.path.join(cfg.BOP_PATH, "bop_toolkit_lib"))
import bop_toolkit_lib.dataset_params as bop_dataset_params
import bop_toolkit_lib.inout as bop_inout
import scipy.io as scio
import trimesh
from environment.renderer import Renderer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(agent, test_loader, dataset_name, bop_results_path=""):
    # --- prepare canonical offsets (ycbv), load models into renderer (verification) and load mask from PoseCNN (ycbv)
    if cfg.USE_CANONICAL:  # adapt to canonical model space
        meta_canon = bop_inout.load_json("../dataset/bop_models_meta.json", True)
        canonical_offsets = {}
        for obj_id in test_loader.dataset.models.keys():
            if obj_id == 0:
                continue
            bop_to_canonical = np.eye(4, dtype=np.float32)
            bop_to_canonical[:3, :3] = np.asarray(meta_canon[obj_id]['R_to_canonical']).reshape(3, 3)
            bop_to_canonical[:3, 3] = np.asarray(meta_canon[obj_id]['t_to_canonical']).squeeze()
            canonical_offsets[obj_id] = bop_to_canonical
    if cfg.USE_VERIFICATION:
        model_params = bop_dataset_params.get_model_params('/'.join(test_loader.dataset.dataset_path.split('/')[:-1]),
                                                           dataset_name)
        mesh_ids = model_params['obj_ids']
        meshes = []
        for mesh_id in mesh_ids:
            mesh = trimesh.load(os.path.join(test_loader.dataset.dataset_path, f"models_eval/obj_{mesh_id:06d}.ply"))
            meshes.append(mesh)
        ren = Renderer(meshes)

        # pre-load estimated masks from PoseCNN for ycbv
        if dataset_name == "ycbv":
            print("loading estimated masks...")
            posecnn_ycbv_path = os.path.join(cfg.POSECNN_YCBV_RESULTS_PATH, "results_PoseCNN_RSS2018")
            ycbv_test_targets = open(os.path.join(posecnn_ycbv_path, "test_data_list.txt"), 'r').readlines()
            ycbv_test_targets = [ycbv_test_target.strip() for ycbv_test_target in ycbv_test_targets]
            mats_ycbv = [scio.loadmat(os.path.join(posecnn_ycbv_path, f"{ycbv_target_id:06d}.mat")) for
                         ycbv_target_id in range(len(ycbv_test_targets))]
            print("done.")
    if bop_results_path != "" and os.path.exists(bop_results_path):
        open(bop_results_path, 'w').write("")  # BOP toolkit expects file to contain results of a single evaluation run

    # --- run evaluation
    agent.eval()
    progress = tqdm(BackgroundGenerator(test_loader), total=len(test_loader))
    with torch.no_grad():
        for data in progress:
            source, target, pose_source, _, scene, critical, _, syms = env.init(data)
            current_source = source.clone()

            # for verification, get observed depth and normal images - load/prepare segmentation masks from PoseCNN
            if cfg.USE_VERIFICATION:
                obj_ids = data['obj_id']
                depths = [test_loader.dataset.get_depth(int(data['scene'][obj_idx]), int(data['frame'][obj_idx]),
                                                        float(data['cam']['depth_scale'][obj_idx]))
                          for obj_idx, obj_id in enumerate(obj_ids)]

                normals = [test_loader.dataset.get_normal(int(data['scene'][obj_idx]), int(data['frame'][obj_idx]),
                                                          float(data['cam']['depth_scale'][obj_idx]))
                           for obj_idx, obj_id in enumerate(obj_ids)]
                # depth/normal should only include mask (for full frame/object)
                if dataset_name == "ycbv":
                    ycbv_target_ids = [ycbv_test_targets.index(
                        f"data/{int(data['scene'][obj_idx]):04d}/{int(data['frame'][obj_idx]):06d}")
                                       for obj_idx, obj_id in enumerate(obj_ids)]
                    masks_vis = []
                    for oi, ycbv_target_id in enumerate(ycbv_target_ids):
                        obj_id = int(data['obj_id'][oi])
                        if obj_id in [19, 20]:  # combine masks for clamps (information lost in single channel image)
                            combined = (mats_ycbv[ycbv_target_id]['labels'] == 19).astype(np.uint8)
                            combined += (mats_ycbv[ycbv_target_id]['labels'] == 20).astype(np.uint8)
                            masks_vis.append((combined > 0).astype(np.uint8))
                        else:
                            masks_vis.append((mats_ycbv[ycbv_target_id]['labels'] == obj_id).astype(np.uint8))
                else:  # dataset_name == "lm"
                    SPLIT = "test"
                    # convert BOP to PoseCNN naming scheme
                    strs_scene = {
                        1: "ape", 2: "benchviseblue", 4: "camera", 5: "can", 6: "cat", 8: "driller", 9: "duck",
                        10: "eggbox", 11: "glue", 12: "holepuncher", 13: "iron", 14: "lamp", 15: "phone"
                    }
                    masks_vis = []
                    for scene_id, im_id in zip(data['scene'], data['frame']):
                        scene_id, im_id = int(scene_id), int(im_id)
                        str_scene = strs_scene[scene_id]
                        posecnn_targets = open(os.path.join(cfg.POSECNN_LM_RESULTS_PATH,
                                                            f"image_set/observed/{str_scene}_{SPLIT}.txt")).read()
                        i_target = posecnn_targets.split().index(f"{scene_id:02d}/{im_id + 1:06d}")

                        # get PoseCNN results
                        mat = scio.loadmat(os.path.join(cfg.POSECNN_LM_RESULTS_PATH, f"{str_scene}/{i_target:04d}.mat"))
                        mask_vis = (mat['labels'] > 0).astype(np.uint8)
                        masks_vis.append(mask_vis)

                for depth, normal, mask_vis in zip(depths, normals, masks_vis):
                    depth[mask_vis == 0] = 0
                    normal[mask_vis == 0] = 0

            # run refinement -- for verification, keep track of intermediary steps
            class_idx = torch.nn.functional.one_hot(data['obj_id'].to(DEVICE) - 1, agent.num_classes)
            step_poses = []
            for step in range(cfg.ITER_EVAL):
                state_emb, action_logit, state_value, _, est_mask = \
                    agent(current_source, target, class_idx=class_idx)
                actions = util_model.action_from_logits(action_logit, deterministic=True)
                current_source, target, pose_source, critical = env.step(source, actions, pose_source, cfg.DISENTANGLED,
                                                                         target, scene)
                if cfg.USE_VERIFICATION:
                    step_poses.append(pose_source.clone()[:, None, ...])

            if cfg.USE_VERIFICATION:
                # determine visual plausibility per object
                scores = np.zeros((len(step_poses[0]), len(step_poses)))
                for oi, obj_id in enumerate(obj_ids):
                    ren.set_observation(depths[oi], normals[oi])
                    # go through refinement steps
                    for si, poses in enumerate(step_poses):
                        predictions_unnorm = poses[:, 0].clone()
                        if cfg.DISENTANGLED:
                            predictions_unnorm = tra.to_global(predictions_unnorm, source).cpu()
                        predictions_unnorm[:, :3, 3] *= data['normalization'][:, 0, 0][:, None]
                        # apply refinement to initial estimate to get the full model-to-camera estimation
                        #   note: prediction is from initial pose to model space
                        init_i2c = data['est_m2c']
                        prediction_m2i = torch.eye(4, device="cpu").repeat(poses.shape[0], 1, 1)
                        prediction_m2i[:, :3, :3] = predictions_unnorm[:, :3, :3].transpose(2, 1)
                        prediction_m2i[:, :3, 3] = -(prediction_m2i[:, :3, :3] @
                                                     predictions_unnorm[:, :3, 3].view(-1, 3, 1)).squeeze()
                        estimates_m2c = init_i2c @ prediction_m2i
                        if cfg.USE_CANONICAL:  # from canonical to BOP model space
                            canonical_offset = torch.FloatTensor(
                                np.stack([canonical_offsets[int(obj_id)] for obj_id in data['obj_id']]))
                            estimates_m2c = estimates_m2c @ canonical_offset
                        # translation in meters
                        pose_meters = estimates_m2c[oi].cpu().numpy().copy()
                        pose_meters[:3, 3] /= 1000
                        score = ren.compute_score([int(obj_id) - 1], [pose_meters],
                                                  np.eye(4), data['cam']['cam_K'][oi].cpu().numpy(),
                                                  cull_back=not(int(obj_id) == 14 and dataset_name == "lm"),
                                                  use_normals=cfg.USE_NORMALS)  # lamp model (id 14) is single-sided
                        scores[oi, si] = score

                # select best refinement step:
                # -> take last step if score is equal for multiple steps (but only if it is != 0 -> not diverging)
                best_steps = np.asarray([[oi, int(np.argwhere(scores[oi] == score_max)[(-1 if score_max > 1e-3 else 0)])]
                                         for oi, score_max in enumerate(scores.max(axis=1))])[:, 1]
                poses = torch.cat(step_poses, dim=1)
                pose_source = poses[torch.arange(pose_source.shape[0]), best_steps[None, :]][0]

            if cfg.DISENTANGLED:
                pose_source = tra.to_global(pose_source, source)
            # undo normalization
            predictions_unnorm = pose_source.clone().cpu()
            predictions_unnorm[:, :3, 3] *= data['normalization'][:, 0, 0][:, None]
            # apply refinement to initial estimate to get the full model-to-camera estimation
            #   note: prediction is from initial pose to model space
            init_i2c = data['est_m2c']
            prediction_m2i = torch.eye(4, device="cpu").repeat(pose_source.shape[0], 1, 1)
            prediction_m2i[:, :3, :3] = predictions_unnorm[:, :3, :3].transpose(2, 1)
            prediction_m2i[:, :3, 3] = -(prediction_m2i[:, :3, :3] @
                                         predictions_unnorm[:, :3, 3].view(-1, 3, 1)).squeeze()
            estimates_m2c = init_i2c @ prediction_m2i
            if cfg.USE_CANONICAL:  # from canonical to BOP model space
                canonical_offset = torch.FloatTensor(np.stack([canonical_offsets[int(obj_id)] for obj_id in data['obj_id']]))
                estimates_m2c = estimates_m2c @ canonical_offset
            # save in BOP format
            estimates_bop = ""
            for i_est, estimate in enumerate(estimates_m2c):
                scene_id, im_id, obj_id = data['scene'][i_est], data['frame'][i_est],\
                                          data['obj_id'][i_est]
                conf, duration = 1.0, 0.0
                estimates_bop += f"{scene_id},{im_id},{obj_id},{conf:0.3f}," \
                                 f"{' '.join([f'{float(v):0.6f}' for v in estimate[:3, :3].reshape(-1)])}," \
                                 f"{' '.join([f'{float(v):0.6f}' for v in estimate[:3, 3].reshape(-1)])}," \
                                 f"{duration:0.3f}\n"
            with open(bop_results_path, 'a') as file:
                file.write(estimates_bop)
    print(f"Stored predictions in BOP format to {bop_results_path}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SporeAgent - evaluation on LINEMOD and YCB-VIDEO')
    parser.add_argument('--mode', type=str, default='ilrl', choices=['il', 'ilrl'],
                        help='IL-only (il), IL+RL (ilrl)')
    parser.add_argument('--dataset', type=str, default='lm', choices=['lm', 'ycbv'],
                        help='Dataset used for evaluation.')

    args = parser.parse_args()
    if args.dataset == "lm":
        cfg.USE_SYMMETRY = False  # not defined for LM
        cfg.USE_CANONICAL = False  # not defined for LM
        cfg.UPDATE_SOURCE_DISTANCE = False  # only plane, so not necessary

    code_path = os.path.dirname(os.path.abspath(__file__)).replace("/registration", "")
    if args.dataset == "lm":
        from dataset.dataset import DatasetLinemod
        test_dataset = DatasetLinemod("test")
    elif args.dataset == "ycbv":
        from dataset.dataset import DatasetYcbVideo
        test_dataset = DatasetYcbVideo("test")

    pretrain = os.path.join(code_path, f"weights/{args.dataset}_{args.mode}.zip")
    if not os.path.exists(os.path.join(code_path, f"results")):
        os.mkdir(os.path.join(code_path, f"results"))
    bop_results_path = os.path.join(code_path, f"results/sporeagent-{args.mode}_{args.dataset}-test.csv")
    print(f"  results path = {bop_results_path}")

    # note: test is deterministic (sets seed to dataset sample's index)
    # lm: 32 x 4traj x 1obj = 128 -vs- ycbv: 8 x 4traj x [num_frame_objects~4] ~ 128
    batch_size = 32 if args.dataset == "lm" else 8
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              collate_fn=dataset.dataset.collate_data, shuffle=False)

    print("  loading weights...")
    agent = Agent().to(DEVICE)
    if os.path.exists(pretrain):
        util_model.load(agent, pretrain)
    else:
        raise FileNotFoundError(f"No weights found at {pretrain}. Download pretrained weights or run training first.")

    evaluate(agent, test_loader, args.dataset, bop_results_path)
