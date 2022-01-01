import numpy as np
import scipy.io as scio
from scipy.spatial.transform.rotation import Rotation
import json
import os
import config as cfg

file = "sporeagent-ilrl"
folder = f"results_{file}"
results_bop = open(f"../results/{file}_ycbv-test.csv", 'r').read().split("\n")
bop_offsets = dict([[int(obj_id), np.asarray(offset)] for obj_id, offset in
                    json.load(open("../dataset/ycbv_offsets.json", 'r')).items()])
scene_id, im_id, obj_id = -1, -1, -1
results_ycb = dict()
for result_bop in results_bop:
    result = result_bop.split(",")
    if len(result) < 7:
        continue
    if scene_id == int(result[0]) and im_id == int(result[1]) and obj_id == int(result[2]):
        # only best per object
        continue
    scene_id = int(result[0])
    im_id = int(result[1])
    obj_id = int(result[2])
    pose = np.eye(4)
    pose[:3, :3] = np.asarray([float(v) for v in result[4].split(" ")]).reshape(3, 3)
    pose[:3, 3] = np.asarray([float(v) for v in result[5].split(" ")]).squeeze()

    # undo BOP pose offset, t to meter (-> to YCBV model space)
    offset = np.eye(4, dtype=np.float32)
    offset[:3, 3] = -bop_offsets[obj_id]
    pose = pose @ offset
    pose[:3, 3] /= 1000.0

    # pose to ycbv format
    q = Rotation.from_matrix(pose[:3, :3]).as_quat().tolist()  # xyzw format
    q_xyz, q_w = q[:-1], [q[-1]]
    q = np.asarray(q_w + q_xyz)
    pose = np.hstack([q, pose[:3, 3]])
    roi = np.array([0, obj_id, 0, 0, 0, 0, 0])

    u_id = f"{scene_id:04d},{im_id:06d}"
    if u_id not in results_ycb:
        results_ycb[u_id] = []
    results_ycb[u_id].append([roi, pose])
u_ids = np.sort(np.unique(list(results_ycb.keys())))

# merge results per image st roi is [num_obj, 7]
if not os.path.exists(os.path.join(cfg.POSECNN_YCBV_RESULTS_PATH, f"{folder}")):
    os.mkdir(os.path.join(cfg.POSECNN_YCBV_RESULTS_PATH, f"{folder}"))
for i, u_id in enumerate(u_ids):
    results = results_ycb[u_id]
    rois = np.vstack([result[0] for result in results])
    poses_ours = np.vstack([result[1] for result in results])
    scio.savemat(os.path.join(cfg.POSECNN_YCBV_RESULTS_PATH, f"{folder}/{i:06d}.mat"),
                 {'rois': rois, 'poses_ours': poses_ours})
