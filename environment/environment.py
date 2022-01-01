import torch
import config as cfg
import environment.transformations as tra
import environment.convexhull as cvx
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""
Encapsulates the refinement environment behavior, i.e., initialization, updating the state given an action, computing
the reward for an action in the current state and, additionally, provides the expert policy's action for a given state. 
"""

ALL_STEPS = torch.FloatTensor(cfg.STEPSIZES[::-1] + [0] + cfg.STEPSIZES).to(DEVICE)
POS_STEPS = torch.FloatTensor([0] + cfg.STEPSIZES).to(DEVICE)
NUM_STEPS = len(cfg.STEPSIZES)


def init(data):
    """
    Get the initial observation, the ground-truth pose for the expert and initialize the agent's accumulator (identity).
    """
    # observation
    channels = 6 if cfg.USE_NORMALS else 3

    pcd_source, pcd_target = data['points_src'][..., :channels].to(DEVICE),\
                             data['points_ref'][..., :channels].to(DEVICE)
    B = pcd_source.shape[0]

    # GT (for expert)
    pose_target = torch.eye(4, device=DEVICE).repeat(B, 1, 1)
    pose_target[:, :3, :] = data['transform_gt']

    # initial estimates (identity, for student)
    pose_source = torch.eye(4, device=DEVICE).repeat(B, 1, 1)

    # optional: augment input by contact distance (est model vs environment)
    if cfg.USE_CONTACT:
        # get objects to plane space
        relative_transformation = torch.inverse(data['relative_plane2points'])  # points(est)->plane(est)
        relative_transformation_norm = torch.inverse(data['plane_normalization']) @ relative_transformation @ data[
            'normalization']  # map between normalized spaces
        distance_scale = data['scale_plane2points'][:, None]  # to normalized object space

        if 'other_obj_ids' in data and cfg.SCENE_REFINE:
            # per object, get the batch index of the other objects in its frame (= potential interactions)
            object_ids = torch.cat([data['scene'][:, None], data['frame'][:, None], data['obj_id'][:, None]],
                                   dim=1).numpy().tolist()
            other_obj_indices = []
            for ti, (scene_id, frame_id, num_frame_objects) in enumerate(
                    zip(data['scene'], data['frame'], data['num_frame_objects'])):
                other_indices = []
                for oi in range(num_frame_objects - 1):
                    other_idx = object_ids.index([scene_id, frame_id, int(data['other_obj_ids'][ti, oi])])
                    other_indices.append(other_idx)
                other_obj_indices.append(other_indices)
        else:
            other_obj_indices = []

        scene = [relative_transformation_norm.to(DEVICE), distance_scale.to(DEVICE), other_obj_indices]

        # add contact information
        target_distances, critical_points, targets_in_plane = get_contacts(scene, pcd_target)
        source_distances, _, targets_in_plane = get_contacts(scene, pcd_source, targets_in_plane=targets_in_plane)
        pcd_source = torch.cat([pcd_source, source_distances[..., None]], dim=2)
        pcd_target = torch.cat([pcd_target, target_distances[..., None]], dim=2)
    else:
        scene = []
        critical_points = torch.zeros((pcd_source.shape[0], pcd_source.shape[1], 2)).to(DEVICE)

    if cfg.USE_SYMMETRY:
        symmetries = data['symmetries'].to(DEVICE)
        # = src->ref -- symmetries are in ref space -> inv(inv(tgt) @ sym) = inv(sym) @ tgt
        pose_target = torch.inverse(symmetries) @ pose_target[:, None, ...]
    else:
        symmetries = torch.eye(4, device=DEVICE).repeat(B, 1, 1, 1)

    # segmentation
    if cfg.USE_SEGMENT:
        gt_mask = data['points_src'][..., -1][..., None].to(DEVICE)
    else:
        gt_mask = torch.ones_like(pcd_source[..., -1])

    return pcd_source, pcd_target, pose_source, pose_target, scene, critical_points, gt_mask, symmetries


def _action_to_step(axis_actions):
    """
    Convert action ids to sign and step size.
    """
    step = ALL_STEPS[axis_actions]
    sign = ((axis_actions - NUM_STEPS >= 0).float() - 0.5) * 2
    return sign, step


def step(source, actions, pose_source, disentangled=True, target=None, scene=None):
    """
    Update the state (source and accumulator) using the given actions.
    """
    actions_t, actions_r = actions[:, 0], actions[:, 1]
    indices = torch.arange(source.shape[0]).unsqueeze(0)

    # actions to transformations
    steps_t = torch.zeros((actions.shape[0], 3), device=DEVICE)
    steps_r = torch.zeros((actions.shape[0], 3), device=DEVICE)
    for i in range(3):
        sign, step = _action_to_step(actions_t[:, i])
        steps_t[indices, i] = step * sign

        sign, step = _action_to_step(actions_r[:, i])
        steps_r[indices, i] = step * sign

    # accumulate transformations
    if disentangled:  # eq. 7 in paper
        pose_source[:, :3, :3] = tra.euler_angles_to_matrix(steps_r, 'XYZ') @ pose_source[:, :3, :3]
        pose_source[:, :3, 3] += steps_t
    else:  # concatenate 4x4 matrices (eq. 5 in paper)
        pose_update = torch.eye(4, device=DEVICE).repeat(pose_source.shape[0], 1, 1)
        pose_update[:, :3, :3] = tra.euler_angles_to_matrix(steps_r, 'XYZ')
        pose_update[:, :3, 3] = steps_t

        pose_source = pose_update @ pose_source

    # update source with the accumulated transformation
    new_source = tra.apply_trafo(source, pose_source, disentangled)

    # optional: update contact distance (est model vs environment)
    if cfg.USE_CONTACT:
        pose_source_glob = tra.to_global(pose_source.clone(), source)

        # plane to target
        new_scene = [scene[0] @ torch.inverse(pose_source_glob), scene[1], scene[2]]
        # add contact information
        distances, critical_points, targets_in_plane = get_contacts(new_scene, target)
        new_target = target.clone()
        new_target[..., -1] = distances

        # update source
        if cfg.UPDATE_SOURCE_DISTANCE:
            source_distances, _, _ = get_contacts(scene, source, targets_in_plane=targets_in_plane)
            new_source[..., -1] = source_distances
    else:
        critical_points = torch.zeros((new_source.shape[0], new_source.shape[1], 2)).to(DEVICE)
        new_target = target

    return new_source, new_target, pose_source, critical_points


def expert(pose_source, targets, mode='steady'):
    """
    Get the expert action in the current state.
    """
    # compute delta, eq. 10 in paper
    if cfg.USE_SYMMETRY:
        delta_t = targets[..., :3, 3] - pose_source[:, None, :3, 3]
        delta_R = targets[..., :3, :3] @ pose_source[:, None, :3, :3].transpose(-1, -2)  # global accumulator
        if cfg.SYMMETRY_BEST:  # -> if we have symmetric alternatives, take the one with the least effort
            # since symmetries depend on rotation (and translation is just adapted accordingly), the magnitude of
            # the required rotation is a good measure for minimal effort -- we use the isotropic error
            delta_R_trace = delta_R[..., 0, 0] + delta_R[..., 1, 1] + delta_R[..., 2, 2]
            delta_R_iso = torch.acos(torch.clamp(0.5 * (delta_R_trace - 1), min=-1.0, max=1.0))
            idx_best = torch.argmin(delta_R_iso.abs(), dim=1)

            indices = torch.arange(delta_t.size(0)).unsqueeze(0)
            delta_t = delta_t[indices, idx_best].reshape(-1, 3)
            delta_R = delta_R[indices, idx_best].reshape(-1, 3, 3)
    else:
        delta_t = targets[:, :3, 3] - pose_source[:, :3, 3]
        delta_R = targets[:, :3, :3] @ pose_source[:, :3, :3].transpose(-1, -2)
    delta_r = tra.matrix_to_euler_angles(delta_R, 'XYZ')

    def _get_axis_action(axis_delta, mode='steady'):
        lower_idx = (torch.bucketize(torch.abs(axis_delta), POS_STEPS) - 1).clamp(0, NUM_STEPS)
        if mode == 'steady':
            nearest_idx = lower_idx
        elif mode == 'greedy':
            upper_idx = (lower_idx + 1).clamp(0, NUM_STEPS)
            lower_dist = torch.abs(torch.abs(axis_delta) - POS_STEPS[lower_idx])
            upper_dist = torch.abs(POS_STEPS[upper_idx] - torch.abs(axis_delta))
            nearest_idx = torch.where(lower_dist < upper_dist, lower_idx, upper_idx)
        else:
            raise ValueError

        # -- step idx to action
        axis_action = nearest_idx  # [0, num_steps] -- 0 = NOP
        axis_action[axis_delta < 0] *= -1  # [-num_steps, num_steps + 1] -- 0 = NOP
        axis_action += NUM_STEPS  # [0, 2 * num_steps + 1 -- num_steps = NOP

        return axis_action[..., None, None]

    # find bounds per axis s.t. b- <= d <= b+
    action_t = torch.cat([_get_axis_action(delta_t[..., i], mode) for i in range(3)], dim=-1)
    action_r = torch.cat([_get_axis_action(delta_r[..., i], mode) for i in range(3)], dim=-1)
    action = torch.cat([action_t, action_r], dim=-2)

    return action


def reward_step(current_pcd_source, gt_pcd_source, prev_chamfer_dist=None):
    """
    Compute the dense step reward for the updated state.
    """
    dist = torch.min(tra.square_distance(current_pcd_source, gt_pcd_source), dim=-1)[0]
    chamfer_dist = torch.mean(dist, dim=1).view(-1, 1, 1)

    if prev_chamfer_dist is not None:
        better = (chamfer_dist < prev_chamfer_dist).float() * 0.5
        same = (chamfer_dist == prev_chamfer_dist).float() * 0.1
        worse = (chamfer_dist > prev_chamfer_dist).float() * 0.6

        reward = better - worse - same
        return reward, chamfer_dist
    else:
        return torch.zeros_like(chamfer_dist), chamfer_dist


def get_contacts(scene, points, targets_in_plane=None, k=10):
    target_to_other, relative_scale, others_indices = scene

    # === 1) all objects into plane space
    points_in_plane = (target_to_other[:, :3, :3] @ points[..., :3].transpose(2, 1)).transpose(2, 1) \
                      + target_to_other[:, :3, 3][:, None, :]

    # === 2) get signed distance to plane
    signed_distance = points_in_plane[..., 2].clone()
    supported = torch.ones_like(signed_distance)

    if len(others_indices) > 0:  # === 3) get signed distance to other objects in the scene
        # remove scaling (via normalization) for rotation of the normal vectors
        if targets_in_plane is None:
            unscaled_rotation = target_to_other[:, :3, :3] * relative_scale[..., None]

            targets_in_plane = points_in_plane.clone()
            normals_in_plane = (unscaled_rotation @ points[..., 3:6].transpose(2, 1)).transpose(2, 1)
            targets_in_plane = torch.cat([targets_in_plane, normals_in_plane], dim=-1)

        batch_signed_distances = []
        batch_support = []
        for b, other_indices in enumerate(others_indices):
            num_other = len(other_indices)

            # get k nearest neighbors in each of the other objects
            distances, nearests = [], []
            for o in other_indices:
                dist, idx = tra.nearest_neighbor(targets_in_plane[o, ..., :3][None, ...],
                                                 points_in_plane[b, ..., :3][None, ...], k=k)
                near = targets_in_plane[o][idx[0]]
                distances.append(dist)
                nearests.append(near[None, ...])
            if num_other == 0:  # add plane distance instead (doesn't change min)
                batch_signed_distances.append(signed_distance[b][None, ...])
                batch_support.append(torch.ones_like(signed_distance[b][None, ...]))
                continue
            distances = torch.cat(distances, dim=0)  # [num_other] x k x N
            nearests = torch.cat(nearests, dim=0)  # [num_other] x k x N x 6

            # check if query is inside or outside based on surface normal
            surface_normals = nearests[..., 3:6]
            gradients = nearests[..., :3] - points_in_plane[b, None, :, :3]  # points towards surface
            gradients = gradients / torch.norm(gradients, dim=-1)[..., None]
            insides = torch.einsum('okij,okij->oki', surface_normals, gradients) > 0  # same direction -> inside
            # filter by quorum of votes
            inside = torch.sum(insides, dim=1) > k * 0.8

            # get nearest neighbor (in each other object)
            distance, gradient, surface_normal = distances[:, 0, ...], gradients[:, 0, ...], surface_normals[:, 0, ...]

            # change sign of distance for points inside
            distance[inside] *= -1

            # take minimum over other points --> minimal SDF overall
            # = the closest outside/farthest inside each point is wrt any environment collider
            if num_other == 1:
                batch_signed_distances.append(distance[0][None, ...])
            else:
                distance, closest = distance.min(dim=0)
                batch_signed_distances.append(distance[None, ...])

            # check whether closest points could support the object (irrespective of in/out)
            # -> if gravity is less than orthogonal to contact surface
            if num_other == 1:
                surface_normals = surface_normals[0]  # k x N x 3
            else:  # only those of closest
                surface_normals = torch.gather(surface_normals, 0,
                                               closest[None, None, :, None].repeat(surface_normals.shape[0], k, 1, 3))[0]

            gravity = torch.zeros_like(surface_normals)
            gravity[..., 2] = -1  # negative z
            # adapting the allowed angle here would change friction; atm it is very high
            supports = torch.einsum('kij,kij->ki', surface_normals, gravity) < 0  # opposite direction -> supporting
            # filter by quorum of votes
            support = torch.sum(supports, dim=0) > k * 0.8
            batch_support.append(support[None, ...])
        signed_distances = torch.cat(batch_signed_distances, dim=0)
        supports = torch.cat(batch_support, dim=0)

        signed_distance, closest = torch.cat([signed_distance[:, None], signed_distances[:, None]], dim=1).min(dim=1)
        supported = torch.cat([supported[:, None],
                               supports[:, None]], dim=1).gather(1,
                                                                 closest[:, None, :].repeat(1, supports.shape[0]+1, 1))[:, 0]
    else:
        supported = torch.ones_like(signed_distance)

    # === 4) derive critical points - allows to determine feasibility and stability
    contacts, intersects = signed_distance.abs() < cfg.TOL_CONTACT, signed_distance < -cfg.TOL_CONTACT
    critical_points = torch.cat([contacts[..., None], intersects[..., None], supported[..., None]], dim=-1)

    # back to normalized scale of the target object (st distance relates to change in xyz coordinates)
    signed_distance *= relative_scale

    return signed_distance, critical_points, targets_in_plane


def is_non_floating(critical_points):
    return (critical_points[..., 0].sum(dim=1) > 0).float()  # at least one contact point


def is_non_intersecting(critical_points):
    return (critical_points[..., 1].sum(dim=1) == 0).float()  # no intersecting points


def is_stable(scene, points, feasible_samples, critical_points):
    # to plane space: z-axis aligned with gravity direction (assumed to be plane normal)
    points_to_plane, relative_scale, others_indices = scene
    points_in_plane = (points_to_plane[:, :3, :3] @ points[..., :3].transpose(2, 1)).transpose(2, 1) \
                      + points_to_plane[:, :3, 3][:, None, :]

    # check stability: need to look at each pcd separately as hull has varying shape
    stables = torch.zeros_like(feasible_samples)
    for bi, (obj_points, feasible, criticals) in enumerate(zip(points_in_plane, feasible_samples, critical_points)):
        if feasible == 1:  # feasibility is a precondition for stability -> guarantees that |contacts| > 0
            # note: this is a simplified test that assumes that the contact points span a polygon (i.e., stable poses
            # on a single point or line are rejected) and friction is always sufficiently large

            supports = (criticals[:, 0] + criticals[:, 2] == 2).bool()  # in contact and supporting surface angle
            if supports.sum() == 0:
                continue
            supporting_points = obj_points[supports]

            # get convex hull of supporting points (support polygon)
            supporting_points_xy = supporting_points[..., :2]  # z-axis is assumed to be aligned with gravity direction
            support_polygon = cvx.quickhull(supporting_points_xy.cpu().numpy())

            # check stability (if CoM is inside support polygon)
            com = points_to_plane[bi][:3, 3]
            stable = 1 if cvx.points_in_hull(com[:2][None, :].cpu().numpy(), support_polygon) else 0
            stables[bi] = stable
    return stables
