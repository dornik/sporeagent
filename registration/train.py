import numpy as np
np.random.seed(42)
import torch
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_dtype(torch.float32)
import torch.nn.functional as F
import os
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
import argparse

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)).replace("/registration", ""))
from environment import environment as env
from environment import transformations as tra
from environment.buffer import Buffer
from registration.model import Agent
import registration.model as util_model
import utility.metrics as metrics
from utility.logger import Logger
import dataset.dataset as dtst
from dataset.dataset import DatasetLinemod, DatasetYcbVideo
import config as cfg
import trimesh

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(agent, logger, dataset, epochs, lr, lr_step, alpha, model_path, reward_mode=""):
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_step, 0.5)

    Dataset = DatasetLinemod if dataset == "lm" else DatasetYcbVideo
    train_dataset = Dataset("train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE,
                                               collate_fn=dtst.collate_data, shuffle=True)
    val_dataset = Dataset("val")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE,
                                             collate_fn=dtst.collate_data, shuffle=False)
    test_dataset = Dataset("eval")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE,
                                              collate_fn=dtst.collate_data, shuffle=False)

    RANDOM_STATE = np.random.get_state()  # otherwise loader produces deterministic samples after iter 1
    losses_bc, losses_ppo, losses_seg, \
    losses_score, losses_floating, losses_intersecting, losses_stable, \
    train_rewards, final_rewards = [], [], [], [], [], [], [], [], []
    episode = 0  # for loss logging (not using epoch)
    best_chamfer = np.infty
    best_vals = dict()

    buffer = Buffer()
    buffer.start_trajectory()
    for epoch in range(epochs):
        print(f"Epoch {epoch}")

        # -- train
        agent.train()
        np.random.set_state(RANDOM_STATE)

        progress = tqdm(BackgroundGenerator(train_loader), total=len(train_loader))
        total_buffered = 0
        for data in progress:
            with torch.no_grad():
                total_buffered += data['scene'].shape[0]

                # per sample, generate a full trajectory
                source, target, pose_source, pose_target, scene, critical, gt_mask, syms = env.init(data)

                if cfg.DISENTANGLED:
                    pose_target = tra.to_disentangled(pose_target, source)
                current_source = source
                if "step" in reward_mode:
                    if cfg.USE_SYMMETRY:
                        gt_pcd_source = tra.apply_trafo(current_source, pose_target[:, 0], disentangled=cfg.DISENTANGLED)
                    else:
                        gt_pcd_source = tra.apply_trafo(current_source, pose_target, disentangled=cfg.DISENTANGLED)
                    _, prev_chamfer = env.reward_step(current_source, gt_pcd_source)

                # -- compute plausibility
                if "stable" in reward_mode:
                    # compute plausibility label
                    non_floating = env.is_non_floating(critical)
                    non_intersecting = env.is_non_intersecting(critical)
                    feasible = ((non_floating + non_intersecting) == 2).float()

                    pose_source_glob = tra.to_global(pose_source.clone(), source)
                    new_scene = [scene[0] @ torch.inverse(pose_source_glob), scene[1], scene[2]]
                    stable = env.is_stable(new_scene, target, feasible, critical)
                    current_plausibility = torch.cat([non_floating[..., None], non_intersecting[..., None],
                                                   stable[..., None]], dim=-1)
                else:  # pretty expensive - do only when needed
                    current_plausibility = torch.zeros_like(target[..., :3])

                # -- one-hot instance vector (for segmentation)
                class_idx = torch.nn.functional.one_hot(data['obj_id'].to(DEVICE) - 1, agent.num_classes)

                # STAGE 1: generate trajectories
                for step in range(cfg.ITER_TRAIN):
                    # expert prediction
                    expert_action = env.expert(pose_source, pose_target, mode=cfg.EXPERT_MODE)

                    # student prediction -- stochastic policy
                    state_emb, action_logit, state_value, _, est_mask = agent(current_source, target, class_idx=class_idx)

                    action = util_model.action_from_logits(action_logit, deterministic=False)
                    action_logprob, action_entropy = util_model.action_stats(action_logit, action)

                    # step environment and get reward
                    new_source, new_target, pose_source, new_critical = env.step(source, action, pose_source,
                                                                                 cfg.DISENTANGLED, target, scene)

                    # -- compute plausibility
                    if "stable" in reward_mode:
                        # compute plausibility label
                        non_floating = env.is_non_floating(new_critical)
                        non_intersecting = env.is_non_intersecting(new_critical)
                        feasible = ((non_floating + non_intersecting) == 2).float()

                        pose_source_glob = tra.to_global(pose_source.clone(), source)
                        new_scene = [scene[0] @ torch.inverse(pose_source_glob), scene[1], scene[2]]
                        stable = env.is_stable(new_scene, target, feasible, new_critical)
                        true_plausibility = torch.cat([non_floating[..., None], non_intersecting[..., None],
                                                       stable[..., None]], dim=-1)
                    else:  # pretty expensive - do only when needed
                        true_plausibility = torch.zeros_like(target[..., :3])

                    # -- compute reward
                    if reward_mode == "step":
                        reward, prev_chamfer = env.reward_step(new_source, gt_pcd_source, prev_chamfer)
                    elif reward_mode == "stable":  # using plausibility after taking this step
                        reward = true_plausibility[:, 2][:, None, None]
                    elif reward_mode == "stable-step":
                        reward, prev_chamfer = env.reward_step(new_source, gt_pcd_source, prev_chamfer)
                        reward += true_plausibility[:, 2][:, None, None] - 0.5
                        reward /= 2
                    else:
                        reward = torch.zeros((pose_source.shape[0], 1, 1)).to(DEVICE)

                    # log trajectory -- using plausibility of input
                    buffer.log_step([current_source, target, current_plausibility, gt_mask, class_idx, syms],
                                    state_value, reward,
                                    expert_action,
                                    action, action_logit, action_logprob)

                    current_source = new_source
                    current_plausibility = true_plausibility
                    if cfg.USE_CONTACT:
                        target = new_target

                    train_rewards.append(reward.view(-1))
                final_rewards.append(reward.view(-1))

            if total_buffered >= 128:
                # STAGE 2: policy (and value estimator) update using BC (and PPO)

                # convert buffer to tensor of samples (also computes return and advantage over trajectories)
                samples = buffer.get_samples()
                ppo_dataset = torch.utils.data.TensorDataset(*samples)
                ppo_loader = torch.utils.data.DataLoader(ppo_dataset, batch_size=cfg.BATCH_SIZE_BUFFER, shuffle=True,
                                                         drop_last=False)

                # sample batches from buffer and update
                for batch in ppo_loader:
                    sources, targets, true_plausibilities, gt_masks, class_indices, symmetries, \
                    expert_actions, state_values, \
                    actions, action_logits, action_logprobs, \
                    returns, advantages = batch

                    # -- predict using current policy
                    new_state_emb, new_action_logit, new_values, _, new_masks = agent(sources, targets, class_idx=class_indices)
                    new_action_logprob, new_action_entropy = util_model.action_stats(new_action_logit, actions)

                    # -- clone term
                    stepsizes = 2 * cfg.NUM_STEPSIZES + 1  # plus, minus, nop
                    loss_translation = F.cross_entropy(new_action_logit[0].view(-1, stepsizes, 1, 1, 1),
                                                       expert_actions[:, 0].reshape(-1, 1, 1, 1))
                    loss_rotation = F.cross_entropy(new_action_logit[1].view(-1, stepsizes, 1, 1, 1),
                                                    expert_actions[:, 1].reshape(-1, 1, 1, 1))
                    clone_loss = (loss_translation + loss_rotation) / 2

                    if alpha > 0:
                        # -- policy term
                        # ratio: lp > prev_lp --> probability of selecting that action increased
                        ratio = torch.exp(new_action_logprob - action_logprobs).view(-1, 6)
                        policy_loss = -torch.min(ratio * advantages.repeat(1, 6),
                                                 ratio.clamp(1 - cfg.CLIP_EPS,
                                                             1 + cfg.CLIP_EPS) * advantages.repeat(1, 6)).mean()

                        # -- value term
                        value_loss = (new_values.view(-1, 1) - returns).pow(2)
                        if cfg.CLIP_VALUE:
                            values_clipped = state_values + (new_values - state_values)\
                                .clamp(-cfg.CLIP_EPS, cfg.CLIP_EPS)
                            losses_v_clipped = (values_clipped.view(-1, 1) - returns).pow(2)
                            value_loss = torch.max(value_loss, losses_v_clipped)
                        value_loss = value_loss.mean()

                        # -- entropy term
                        entropy_loss = new_action_entropy.mean()

                    # -- segmentation term
                    if cfg.USE_SEGMENT:
                        seg_loss = F.binary_cross_entropy(new_masks.view(-1, 1024), gt_masks.view(-1, 1024))

                    # -- update
                    optimizer.zero_grad()
                    loss = clone_loss
                    losses_bc.append(clone_loss.item())
                    if alpha > 0:
                        ppo_loss = policy_loss + value_loss * cfg.C_VALUE - entropy_loss * cfg.C_ENTROPY
                        loss += ppo_loss * alpha
                        losses_ppo.append(ppo_loss.item())
                    if cfg.USE_SEGMENT:
                        loss += seg_loss * cfg.C_SEGMENT
                        losses_seg.append(seg_loss.item())
                    loss.backward()
                    optimizer.step()

                # logging
                if alpha > 0:
                    logger.record("train/ppo", np.mean(losses_ppo))
                if cfg.USE_SEGMENT:
                    logger.record("train/seg", np.mean(losses_seg))
                logger.record("train/bc", np.mean(losses_bc))
                logger.record("train/reward", float(torch.cat(train_rewards, dim=0).mean()))
                logger.record("train/final_reward", float(torch.cat(final_rewards, dim=0).mean()))
                logger.dump(step=episode)

                # reset
                losses_bc, losses_ppo, losses_seg,\
                losses_score, losses_floating, losses_intersecting, losses_stable,\
                train_rewards, final_rewards = [], [], [], [], [], [], [], [], []
                buffer.clear()
                total_buffered = 0
                episode += 1

            buffer.start_trajectory()
        scheduler.step()
        RANDOM_STATE = np.random.get_state()  # evaluation sets seeds again -- keep random state of the training stage

        # -- test
        if val_loader is not None:
            mtrcs = evaluate(agent, logger, val_loader, prefix='val')
        if test_loader is not None:
            mtrcs = evaluate(agent, logger, test_loader)

        if isinstance(mtrcs, dict):
            for mtrc, val in mtrcs.items():
                if mtrc not in best_vals:
                    best_vals[mtrc] = val
                elif val <= best_vals[mtrc]:
                    best_vals[mtrc] = val
                else:
                    continue
                print(f"new best {mtrc}: {val}")
                infos = {
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                util_model.save(agent, f"{model_path}_{mtrc}.zip", infos)
        elif mtrcs <= best_chamfer:
            print(f"new best: {mtrcs}")
            best_chamfer = mtrcs
            infos = {
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
            }
            util_model.save(agent, f"{model_path}.zip", infos)
        logger.dump(step=epoch)


def evaluate(agent, logger, loader, prefix='test'):
    agent.eval()
    progress = tqdm(BackgroundGenerator(loader), total=len(loader))
    predictions = []
    val_losses = []
    with torch.no_grad():
        for data in progress:
            source, target, pose_source, pose_target, scene, critical, _, syms = env.init(data)
            if cfg.DISENTANGLED:
                pose_target = tra.to_disentangled(pose_target, source)

            # -- one-hot instance vector (for segmentation)
            class_idx = torch.nn.functional.one_hot(data['obj_id'].to(DEVICE) - 1, agent.num_classes)

            current_source = source
            for step in range(cfg.ITER_EVAL):
                expert_action = env.expert(pose_source, pose_target, mode=cfg.EXPERT_MODE)

                state_emb, action_logit, _, _, est_mask = agent(current_source, target, class_idx=class_idx)
                action = util_model.action_from_logits(action_logit, deterministic=True)

                stepsizes = 2 * cfg.NUM_STEPSIZES + 1  # plus, minus, nop
                loss_translation = F.cross_entropy(action_logit[0].view(-1, stepsizes, 1, 1, 1),
                                                   expert_action[:, 0].reshape(-1, 1, 1, 1))
                loss_rotation = F.cross_entropy(action_logit[1].view(-1, stepsizes, 1, 1, 1),
                                                expert_action[:, 1].reshape(-1, 1, 1, 1))
                val_losses.append((loss_translation + loss_rotation).item()/2)

                current_source, target, pose_source, critical = env.step(source, action, pose_source, cfg.DISENTANGLED,
                                                                         target, scene)
            if cfg.DISENTANGLED:
                pose_source = tra.to_global(pose_source, source)
            predictions.append(pose_source)

    predictions = torch.cat(predictions)
    _, summary_metrics = metrics.compute_stats(predictions, data_loader=loader)

    # log test metrics
    if cfg.USE_SYMMETRY:
        logger.record(f"{prefix}/add", summary_metrics['add'])
        logger.record(f"{prefix}/adi", summary_metrics['adi'])
        logger.record(f"{prefix}/mssd", summary_metrics['mssd'])
        return summary_metrics['add']
    else:
        logger.record(f"{prefix}/add", summary_metrics['add'])
        logger.record(f"{prefix}/adi", summary_metrics['adi'])
        return summary_metrics['add']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SporeAgent - training on LINEMOD and YCB-VIDEO')
    parser.add_argument('--mode', type=str, default='ilrl', choices=['il', 'ilrl'],
                        help='IL-only (il), IL+RL (ilrl).')
    parser.add_argument('--dataset', type=str, default='lm', choices=['lm', 'ycbv'],
                        help='Dataset used for training.')
    args = parser.parse_args()
    if args.dataset == "lm":
        cfg.USE_SYMMETRY = False  # not defined for LM
        cfg.USE_CANONICAL = False  # not defined for LM
        cfg.UPDATE_SOURCE_DISTANCE = False  # only plane, so not necessary

    # PATHS
    dataset = args.dataset
    mode = args.mode
    code_path = os.path.dirname(os.path.abspath(__file__)).replace("/registration", "")
    if not os.path.exists(os.path.join(code_path, "logs")):
        os.mkdir(os.path.join(code_path, "logs"))
    if not os.path.exists(os.path.join(code_path, "weights")):
        os.mkdir(os.path.join(code_path, "weights"))
    model_path = os.path.join(code_path, f"weights/{dataset}_{mode}")
    logger = Logger(log_dir=os.path.join(code_path, f"logs/{dataset}/"), log_name=f"{mode}",
                    reset_num_timesteps=True)

    # TRAINING
    agent = Agent().to(DEVICE)

    if args.mode == "il":
        alpha = 0.0
        reward_mode = ""
    else:
        alpha = 0.1 if dataset == "lm" else 0.2
        reward_mode = "stable-step"
    print(f"Training: dataset '{dataset}' - mode '{args.mode}'{f' - alpha={alpha}' if args.mode != 'il' else ''}")

    if dataset == "ycbv":
        cfg.BATCH_SIZE = 8  # 32 x 4traj x 1obj = 128 -vs- 8 x 4traj x [num_frame_objects~4] ~ 128
    epochs = 100
    lr = 1e-3
    lr_step = 20

    train(agent, logger, dataset, epochs=epochs, lr=lr, lr_step=lr_step,
          alpha=alpha, reward_mode=reward_mode, model_path=model_path)
