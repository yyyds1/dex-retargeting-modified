import tempfile
from pathlib import Path
from typing import Optional, Tuple, List
import csv

import numpy as np
import tyro
import sapien
import torch
from pytransform3d import transformations as pt

from dex_retargeting import yourdfpy as urdf
from dataset import DexYCBVideoDataset
from dex_retargeting.constants import RobotName, HandType, get_default_config_path, RetargetingType
from mano_layer import MANOLayer
from dex_retargeting.retargeting_config import RetargetingConfig

# For numpy version compatibility
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.unicode_

ROBOT_JOINT_TRAJ_DIR = "./traj/"


def compute_hand_geometry(hand_pose_frame, extrinsic_mat, mano_layer, use_camera_frame=False):
    # pose parameters all zero, no hand is detected
    if np.abs(hand_pose_frame).sum() < 1e-5:
        return None, None
    p = torch.from_numpy(hand_pose_frame[:, :48].astype(np.float32))
    t = torch.from_numpy(hand_pose_frame[:, 48:51].astype(np.float32))
    vertex, joint = mano_layer(p, t)
    vertex = vertex.cpu().numpy()[0]
    joint = joint.cpu().numpy()[0]
    if not use_camera_frame:
        pose_vec = pt.pq_from_transform(extrinsic_mat)
        camera_pose = sapien.Pose(pose_vec[0:3], pose_vec[3:7]).inv()
        camera_mat = camera_pose.to_transformation_matrix()
        vertex = vertex @ camera_mat[:3, :3].T + camera_mat[:3, 3]
        vertex = np.ascontiguousarray(vertex)
        joint = joint @ camera_mat[:3, :3].T + camera_mat[:3, 3]
        joint = np.ascontiguousarray(joint)

    return vertex, joint


def get_joint_traj(robots: Optional[List[RobotName]], data_root: Path, data_indices: List[int], fps: int):
    dataset = DexYCBVideoDataset(data_root, hand_type="right")

    # Data ID, feel free to change it to visualize different trajectory
    print(len(dataset))
    data_id = 0

    data = dataset[data_id]
    hand_pose = data["hand_pose"]
    hand_shape = data["hand_shape"]
    object_pose = data["object_pose"]
    extrinsic_mat = data["extrinsics"]
    num_ycb_objects = len(data["ycb_ids"])
    mano_layer = MANOLayer("right", hand_shape.astype(np.float32))
    num_frame = hand_pose.shape[0]
    json_data = {}

    # Skip frames where human hand is not detected in DexYCB dataset
    start_frame = 0
    for i in range(0, num_frame):
        init_hand_pose_frame = hand_pose[i]
        vertex, joint = compute_hand_geometry(init_hand_pose_frame, extrinsic_mat, mano_layer)
        if vertex is not None:
            start_frame = i
            break

    for robot_name in robots:
        config_path = get_default_config_path(robot_name, RetargetingType.position, HandType.right)

        # Add 6-DoF dummy joint at the root of each robot to make them move freely in the space
        override = dict(add_dummy_free_joint=True)
        config = RetargetingConfig.load_from_file(config_path, override=override)
        retargeting = config.build()
        robot_file_name = Path(config.urdf_path).stem

        # Build robot
        urdf_path = Path(config.urdf_path)
        if "glb" not in urdf_path.stem:
            urdf_path = urdf_path.with_stem(urdf_path.stem + "_glb")
        robot_urdf = urdf.URDF.load(str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False)
        urdf_name = urdf_path.name
        temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
        temp_path = f"{temp_dir}/{urdf_name}"
        robot_urdf.write_xml_file(temp_path)

        # Retargeting
        traj = []
        pose = []
        # traj.append(retargeting.optimizer.target_joint_names)
        json_data["joint_names"] = retargeting.optimizer.target_joint_names
        for i in range(start_frame, num_frame):
            hand_pose_frame = hand_pose[i]
            vertex, joint = compute_hand_geometry(hand_pose_frame, extrinsic_mat, mano_layer)
            indices = retargeting.optimizer.target_link_human_indices
            ref_value = joint[indices, :]
            retargeting.retarget(ref_value)
            qpos = retargeting.last_qpos
            traj.append(qpos[6:].tolist())
            dim_6_pose = qpos[:6].tolist()
            from scipy.spatial.transform import Rotation as R
            import math

            rotation = R.from_euler("zyx", [dim_6_pose[5], dim_6_pose[4], dim_6_pose[3] + math.pi / 2.0], degrees=False)
            # rotation = R.from_euler("zyx", [0, 0, 0], degrees=False)
            quaternion = rotation.as_quat().tolist()
            pose.append(dim_6_pose[:3] + quaternion)

        json_data["traj"] = traj
        json_data["pose"] = pose

    # json_data["obj_pose_p"] = []
    # json_data["obj_pose_q"] = []
    json_data["obj_pose"] = []
    pose_vec = pt.pq_from_transform(extrinsic_mat)
    camera_pose = sapien.Pose(pose_vec[0:3], pose_vec[3:7]).inv()
    # json_data["obj_list"] = data["ycb_ids"]

    moved_obj_ids = []
    # remove the unchanged object pose
    for obj_id in range(num_ycb_objects):
        start_obj_pose = object_pose[start_frame][obj_id]
        start_obj_pose = camera_pose * sapien.Pose(
        # start_obj_pose = sapien.Pose(
            start_obj_pose[4:], np.concatenate([start_obj_pose[3:4], start_obj_pose[:3]])
        )
        last_obj_pose = object_pose[num_frame - 1][obj_id]
        last_obj_pose = camera_pose * sapien.Pose(
            last_obj_pose[4:], np.concatenate([last_obj_pose[3:4], last_obj_pose[:3]])
        )

        if not (
            np.linalg.norm(start_obj_pose.p - last_obj_pose.p) < 1e-5
            and np.linalg.norm(start_obj_pose.q - last_obj_pose.q) < 1e-5
        ):
            moved_obj_ids.append(obj_id)

    assert len(moved_obj_ids) == 1

    moved_obj_file_path = data["object_mesh_file"][moved_obj_ids[0]]
    json_data["obj_name"] = moved_obj_file_path.split("/")[-2]
    obj_id = moved_obj_ids[0]
    for i in range(start_frame, num_frame):
        object_pose_frame = object_pose[i]
        pos_quat = object_pose_frame[obj_id]
        # apply pi/2 rotation to x axis
        # pos_quat[:3] = np.array([pos_quat[2], pos_quat[1], pos_quat[0]])
        # Quaternion convention: xyzw -> wxyz
        pose = camera_pose * sapien.Pose(pos_quat[4:], np.concatenate([pos_quat[3:4], pos_quat[:3]]))
        # pmose = camera_pose * sapien.Pose(pos_quat[4:], pos_quat[:4])
        # pose = camera_pose * sapien.Pose(pos_quat[4:], np.array([pos_quat[3:4][0], pos_quat[2], pos_quat[1], pos_quat[0]]))
        
        # def quaternion_multiply(q1, q2):
        #     w1, x1, y1, z1 = q1
        #     w2, x2, y2, z2 = q2
        #     w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        #     x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        #     y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        #     z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        #     return np.array([w, x, y, z])

        # rotation_quaternion_y = np.array([np.sqrt(2)/2, 0, -np.sqrt(2)/2, 0])
        # rotation_quaternion_x = np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0, 0])
        # pose.q = quaternion_multiply(rotation_quaternion_y, pose.q)
        # pose.q = quaternion_multiply(rotation_quaternion_x, pose.q)

        # convert pose to array that can be json serialized
        # pose_p = np.array(pose.p).tolist()
        # pose_q = np.array(pose.q).tolist()
        pose = np.concatenate([np.array(pose.p).tolist(), np.array(pose.q).tolist()]).tolist()
        # pose = np.concatenate([np.array(pose.p).tolist(), np.array(pose.q[1:]).tolist(), np.array(pose.q[0:1]).tolist()]).tolist()

        # json_data["obj_pose_p"].append(pose_p)
        # json_data["obj_pose_q"].append(pose_q)
        json_data["obj_pose"].append(pose)

    # save json_data as json file that is easy to read
    file_name = robot_name.name + "_" + data["capture_name"] + ".json"
    with open(ROBOT_JOINT_TRAJ_DIR + file_name, "w") as file:
        import json

        json.dump(json_data, file, indent=2)

        # Save the traj as csv file
        # file_name = robot_name.name + "_" + data["capture_name"] + ".csv"
        # with open(ROBOT_JOINT_TRAJ_DIR + file_name, 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(traj)

        # save the traj as npy file
        # file_name = robot_name.name + "_" + data["capture_name"] + ".npy"
        # np.save(ROBOT_JOINT_TRAJ_DIR + file_name, np.array(traj))
        # print(f"Saved {file_name}")


def main(dexycb_dir: str, robots: Optional[List[RobotName]] = None, data_indices: List[int] = None, fps: int = 10):
    """
    Retarget the human trajectories for grasping object inside DexYCB dataset to the robot.
    The robot trajectories were saved as csv file in 'traj' directory

    Args:
        dexycb_dir: Data root path to the dexycb dataset
        robots: The names of robots to retarget
        fps: frequency to render hand-object trajectory

    """

    data_root = Path(dexycb_dir).absolute()
    robot_dir = Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(robot_dir)
    if not data_root.exists():
        raise ValueError(f"Path to DexYCB dir: {data_root} does not exist.")
    else:
        print(f"Using DexYCB dir: {data_root}")

    get_joint_traj(robots, data_root, data_indices, fps)


if __name__ == "__main__":
    tyro.cli(main)