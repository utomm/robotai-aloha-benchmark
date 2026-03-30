import collections

import mujoco
import numpy as np
from dm_control.suite import base

from gym_aloha.constants import (
    START_ARM_POSE,
    normalize_puppet_gripper_position,
    normalize_puppet_gripper_velocity,
    unnormalize_puppet_gripper_position,
)

BOX_POSE = [None]  # to be changed from outside
TABLE_SIZE = [None]  # to be changed from outside
PICKBLOCK_RANDOMIZATION = [None]  # to be changed from outside
NUM_QUAD_DISTRACTORS = 5

"""
Environment for simulated robot bi-manual manipulation, with joint position control
Action space:      [left_arm_qpos (6),             # absolute joint position
                    left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                    right_arm_qpos (6),            # absolute joint position
                    right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                    left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                    right_arm_qpos (6),         # absolute joint position
                                    right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                    "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                    left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                    right_arm_qvel (6),         # absolute joint velocity (rad)
                                    right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                    "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
"""


class BimanualViperXTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        left_arm_action = action[:6]
        right_arm_action = action[7 : 7 + 6]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[7 + 6]

        left_gripper_action = unnormalize_puppet_gripper_position(normalized_left_gripper_action)
        right_gripper_action = unnormalize_puppet_gripper_position(normalized_right_gripper_action)

        full_left_gripper_action = [left_gripper_action, -left_gripper_action]
        full_right_gripper_action = [right_gripper_action, -right_gripper_action]

        env_action = np.concatenate(
            [left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action]
        )
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [normalize_puppet_gripper_position(left_qpos_raw[6])]
        right_gripper_qpos = [normalize_puppet_gripper_position(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [normalize_puppet_gripper_velocity(left_qvel_raw[6])]
        right_gripper_qvel = [normalize_puppet_gripper_velocity(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(physics)
        obs["qvel"] = self.get_qvel(physics)
        obs["env_state"] = self.get_env_state(physics)
        obs["images"] = {}
        obs["images"]["top"] = physics.render(height=480, width=640, camera_id="top")
        obs["images"]["angle"] = physics.render(height=480, width=640, camera_id="angle")
        obs["images"]["vis"] = physics.render(height=480, width=640, camera_id="front_close")

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class TransferCubeTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table:  # lifted
            reward = 2
        if touch_left_gripper:  # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table:  # successful transfer
            reward = 4
        return reward


class InsertionTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7 * 2 :] = BOX_POSE[0]  # two objects
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_left_gripper = (
            ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
            or ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
            or ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
            or ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        )

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = (
            ("socket-1", "table") in all_contact_pairs
            or ("socket-2", "table") in all_contact_pairs
            or ("socket-3", "table") in all_contact_pairs
            or ("socket-4", "table") in all_contact_pairs
        )
        peg_touch_socket = (
            ("red_peg", "socket-1") in all_contact_pairs
            or ("red_peg", "socket-2") in all_contact_pairs
            or ("red_peg", "socket-3") in all_contact_pairs
            or ("red_peg", "socket-4") in all_contact_pairs
        )
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper:  # touch both
            reward = 1
        if (
            touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table)
        ):  # grasp both
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table):  # peg and socket touching
            reward = 3
        if pin_touched:  # successful insertion
            reward = 4
        return reward


class PickBlockTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position

        def set_free_joint_qpos(joint_name, pose):
            joint_id = physics.model.name2id(joint_name, "joint")
            qpos_adr = physics.model.jnt_qposadr[joint_id]
            physics.data.qpos[qpos_adr : qpos_adr + 7] = pose

        def set_quad_mesh_vertices(mesh_name, corners, thickness):
            mesh_id = mujoco.mj_name2id(physics.model.ptr, mujoco.mjtObj.mjOBJ_MESH, mesh_name)
            vert_start = physics.model.ptr.mesh_vertadr[mesh_id]
            vert_num = physics.model.ptr.mesh_vertnum[mesh_id]

            half_t = thickness / 2.0
            bottom = corners.copy()
            top = corners.copy()
            bottom[:, 2] = -half_t
            top[:, 2] = half_t
            prism_vertices = np.vstack([bottom, top])

            physics.model.ptr.mesh_vert[vert_start : vert_start + vert_num] = prism_vertices

        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)

            assert BOX_POSE[0] is not None
            set_free_joint_qpos("red_box_joint", BOX_POSE[0])

            # Set table size if provided
            if TABLE_SIZE[0] is not None:
                physics.named.model.geom_size["table"] = TABLE_SIZE[0]

            randomization = PICKBLOCK_RANDOMIZATION[0]
            if randomization is not None:
                physics.named.model.geom_rgba["red_box"] = randomization["cube_rgba"]
                physics.named.model.geom_rgba["table"] = randomization["table_rgba"]

                physics.named.model.geom_size["red_box"] = randomization["cube_half_size"]
                physics.named.model.body_mass["box"] = randomization["cube_mass"]
                physics.named.model.body_inertia["box"] = randomization["cube_inertia"]

                for idx in range(NUM_QUAD_DISTRACTORS):
                    joint_name = f"distractor_quad_joint_{idx}"
                    geom_name = f"distractor_quad_{idx}"
                    mesh_name = f"quad_mesh_{idx}"
                    physics.named.model.geom_rgba[geom_name] = randomization["quad_distractor_rgba"][idx]

                    is_active = bool(randomization["quad_distractor_active"][idx])
                    physics.named.model.geom_contype[geom_name] = 1 if is_active else 0
                    physics.named.model.geom_conaffinity[geom_name] = 1 if is_active else 0

                    set_quad_mesh_vertices(
                        mesh_name,
                        randomization["quad_distractor_vertices"][idx],
                        randomization["quad_distractor_thickness"][idx],
                    )

                mujoco.mj_setConst(physics.model.ptr, physics.data.ptr)

                set_free_joint_qpos("red_box_joint", randomization["box_pose"])
                for idx in range(NUM_QUAD_DISTRACTORS):
                    joint_name = f"distractor_quad_joint_{idx}"
                    set_free_joint_qpos(joint_name, randomization["quad_distractor_poses"][idx])
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return reward based on picking up the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        # Get box z-position (use body name "box", not geom name "red_box")
        box_z = physics.named.data.xpos["box"][2]
        # Table height is approximately 0, box should be lifted 0.1m (10cm) above
        lifted_high_enough = box_z > 0.1

        reward = 0
        if touch_right_gripper or touch_left_gripper:
            reward = 1
        if (touch_right_gripper or touch_left_gripper) and not touch_table:  # lifted
            reward = 2
        if lifted_high_enough and not touch_table:  # lifted by 10cm
            reward = 4
        return reward
