import numpy as np


NUM_QUAD_DISTRACTORS = 5


def sample_box_pose(seed=None):
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    rng = np.random.RandomState(seed)

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = rng.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose(seed=None):
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    rng = np.random.RandomState(seed)

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = rng.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = rng.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


def sample_table_size(seed=None):
    # Table size: [half_width, half_depth, half_thickness]
    # Currently fixed, but can be randomized in the future
    half_width = 0.7
    half_depth = 0.5
    thickness = 0.02

    return np.array([half_width, half_depth, thickness])


def _sample_color(rng, low=0.05, high=0.95):
    return rng.uniform(low, high, size=3)


def _sample_color_with_distance(rng, reference_colors, min_distance, max_trials=200):
    for _ in range(max_trials):
        color = _sample_color(rng)
        distances = [np.linalg.norm(color - ref) for ref in reference_colors]
        if all(distance >= min_distance for distance in distances):
            return color

    if len(reference_colors) == 0:
        return _sample_color(rng)

    reference = np.mean(np.asarray(reference_colors), axis=0)
    fallback = np.clip(1.0 - reference, 0.05, 0.95)
    return fallback


def _sample_free_body_xy(
    rng,
    x_range,
    y_range,
    existing_xy,
    min_distance,
    max_trials=200,
):
    x_low, x_high = x_range
    y_low, y_high = y_range

    for _ in range(max_trials):
        candidate = np.array([rng.uniform(x_low, x_high), rng.uniform(y_low, y_high)])
        if all(np.linalg.norm(candidate - point) >= min_distance for point in existing_xy):
            return candidate

    return np.array([
        rng.uniform(x_low, x_high),
        rng.uniform(y_low, y_high),
    ])


def _sample_quad_vertices(rng):
    for _ in range(200):
        width = rng.uniform(0.05, 0.10)
        height = rng.uniform(0.03, 0.07)

        bottom_left = np.array([-width / 2.0, -height / 2.0, 0.0])
        bottom_right = np.array([width / 2.0, -height / 2.0, 0.0])

        top_y = height / 2.0 + rng.uniform(-0.008, 0.008)
        top_left = np.array(
            [
                -width * rng.uniform(0.15, 0.55),
                top_y,
                0.0,
            ]
        )
        top_right = np.array(
            [
                width * rng.uniform(0.18, 0.60),
                top_y + rng.uniform(-0.008, 0.008),
                0.0,
            ]
        )

        vertices = np.vstack([bottom_left, bottom_right, top_right, top_left])
        vertices[:, :2] += rng.uniform(-0.004, 0.004, size=(4, 2))

        e_bottom = vertices[1, :2] - vertices[0, :2]
        e_top = vertices[2, :2] - vertices[3, :2]
        e_left = vertices[3, :2] - vertices[0, :2]
        e_right = vertices[2, :2] - vertices[1, :2]

        cross_1 = np.cross(e_bottom, vertices[2, :2] - vertices[1, :2])
        cross_2 = np.cross(vertices[2, :2] - vertices[1, :2], vertices[3, :2] - vertices[2, :2])
        cross_3 = np.cross(vertices[3, :2] - vertices[2, :2], vertices[0, :2] - vertices[3, :2])
        cross_4 = np.cross(vertices[0, :2] - vertices[3, :2], vertices[1, :2] - vertices[0, :2])
        area = abs(np.cross(e_bottom, vertices[3, :2] - vertices[0, :2]))

        if area < 0.001:
            continue

        if not (cross_1 > 0 and cross_2 > 0 and cross_3 > 0 and cross_4 > 0):
            continue

        bottom_top_parallel = abs(np.cross(e_bottom, e_top)) / (
            np.linalg.norm(e_bottom) * np.linalg.norm(e_top) + 1e-8
        )
        left_right_parallel = abs(np.cross(e_left, e_right)) / (
            np.linalg.norm(e_left) * np.linalg.norm(e_right) + 1e-8
        )

        if bottom_top_parallel < 0.12:
            continue
        if left_right_parallel < 0.12:
            continue

        vertices[:, 2] = 0.0
        return vertices

    fallback = np.array(
        [
            [-0.05, -0.03, 0.0],
            [0.05, -0.02, 0.0],
            [0.028, 0.03, 0.0],
            [-0.04, 0.02, 0.0],
        ]
    )
    return fallback


def _quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    half = angle / 2.0
    s = np.sin(half)
    return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s])


def _quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def sample_pickblock_episode(seed=None):
    rng = np.random.RandomState(seed)

    table_size = sample_table_size(seed)

    cube_scale = rng.uniform(0.75, 1.25)
    cube_half_size = np.array([0.02, 0.02, 0.02]) * cube_scale

    base_mass = 0.05
    base_inertia = np.array([0.002, 0.002, 0.002])
    cube_mass = base_mass * (cube_scale**3)
    cube_inertia = base_inertia * (cube_scale**5)

    cube_position = np.array(
        [
            rng.uniform(-0.08, 0.25),
            rng.uniform(0.35, 0.68),
            0.05,
        ]
    )
    cube_yaw = rng.uniform(-np.pi, np.pi)
    cube_quat = _quat_from_axis_angle([0.0, 0.0, 1.0], cube_yaw)
    box_pose = np.concatenate([cube_position, cube_quat])

    cube_color = _sample_color(rng)
    table_color = _sample_color_with_distance(rng, [cube_color], min_distance=0.35)

    num_distractors = int(rng.randint(0, NUM_QUAD_DISTRACTORS + 1))
    distractor_indices = np.arange(NUM_QUAD_DISTRACTORS)
    rng.shuffle(distractor_indices)
    active_indices = set(distractor_indices[:num_distractors].tolist())
    occupied_xy = [cube_position[:2], np.array([0.0, 0.0]), np.array([0.0, 0.6])]
    hidden_pose = np.array([0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0])

    distractor_poses = []
    distractor_colors = []
    distractor_vertices = []
    distractor_thickness = []
    distractor_active = []

    color_refs = [cube_color, table_color]

    for idx in range(NUM_QUAD_DISTRACTORS):
        is_active = idx in active_indices
        distractor_active.append(is_active)

        if not is_active:
            distractor_poses.append(hidden_pose.copy())
            distractor_colors.append(np.array([0.2, 0.2, 0.2, 0.0]))
            distractor_vertices.append(_sample_quad_vertices(rng))
            distractor_thickness.append(rng.uniform(0.05, 0.1))
            continue

        quad_xy = _sample_free_body_xy(
            rng,
            x_range=(-0.24, 0.24),
            y_range=(0.34, 0.86),
            existing_xy=occupied_xy,
            min_distance=0.10,
        )
        occupied_xy.append(quad_xy)

        quad_yaw = rng.uniform(-np.pi, np.pi)
        yaw_quat = _quat_from_axis_angle([0.0, 0.0, 1.0], quad_yaw)
        face_up_base_quat = _quat_from_axis_angle([0.0, 1.0, 0.0], -np.pi / 2.0)
        quad_quat = _quat_multiply(yaw_quat, face_up_base_quat)
        quad_quat = quad_quat / (np.linalg.norm(quad_quat) + 1e-12)
        quad_pose = np.array([
            quad_xy[0],
            quad_xy[1],
            0.05,
            quad_quat[0],
            quad_quat[1],
            quad_quat[2],
            quad_quat[3],
        ])

        quad_color = _sample_color_with_distance(rng, color_refs, min_distance=0.30)
        color_refs.append(quad_color)

        distractor_poses.append(quad_pose)
        distractor_colors.append(np.concatenate([quad_color, [1.0]]))
        distractor_vertices.append(_sample_quad_vertices(rng))
        distractor_thickness.append(rng.uniform(0.05, 0.1))

    return {
        "box_pose": box_pose,
        "table_size": table_size,
        "cube_rgba": np.concatenate([cube_color, [1.0]]),
        "table_rgba": np.concatenate([table_color, [1.0]]),
        "cube_half_size": cube_half_size,
        "cube_mass": cube_mass,
        "cube_inertia": cube_inertia,
        "num_quad_distractors": num_distractors,
        "quad_distractor_poses": np.asarray(distractor_poses),
        "quad_distractor_rgba": np.asarray(distractor_colors),
        "quad_distractor_vertices": np.asarray(distractor_vertices),
        "quad_distractor_thickness": np.asarray(distractor_thickness),
        "quad_distractor_active": np.asarray(distractor_active, dtype=bool),
    }
