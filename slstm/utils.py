import math
import random
import numpy as np
from trajnetplusplustools import TrackRow


def drop_distant(xyv, r=10):
    """
    Drops pedestrians more than r meters away from primary ped
    """
    distance_2 = np.sum(np.square(xyv[:, :, :2] - xyv[:, 0:1, :2]), axis=2)
    mask = np.nanmin(distance_2, axis=0) < r ** 2
    return xyv[:, mask], mask


def rotate_path(path, theta):
    ct = math.cos(theta)
    st = math.sin(theta)

    return [TrackRow(r.frame, r.car_id, ct * r.x + st * r.y, -st * r.x + ct * r.y,
                     ct*r.xVelocity + st * r.yVelocity, -st * r.xVelocity + ct * r.yVelocity
                     ) for r in path]


def random_rotation_of_paths(paths):
    theta = random.random() * 2.0 * math.pi
    return [rotate_path(path, theta) for path in paths]


def random_rotation(xyv, goals=None):
    theta = random.random() * 2.0 * math.pi
    ct = math.cos(theta)
    st = math.sin(theta)
    r = np.array([[ct, st], [-st, ct]])

    # 旋转位置
    rotated_positions = np.einsum('ptc,ci->pti', xyv[:, :, :2], r)

    # 旋转速度
    rotated_velocities = np.einsum('ptc,ci->pti', xyv[:, :, 2:], r)

    # 合并旋转后的位置和速度
    rotated_xyv = np.concatenate((rotated_positions, rotated_velocities), axis=2)

    if goals is None:
        return rotated_xyv
    else:
        # 如果提供了goals，则旋转goals（假设它们也是[x, y, xVelocity, yVelocity]格式的）
        rotated_goals_pos = np.einsum('tc,ci->ti', goals[:, :2], r)
        rotated_goals_vel = np.einsum('tc,ci->ti', goals[:, 2:], r)
        rotated_goals = np.concatenate((rotated_goals_pos, rotated_goals_vel), axis=1)
        return rotated_xyv, rotated_goals


def theta_rotation(xyv, theta):
    ct = math.cos(theta)
    st = math.sin(theta)
    r = np.array([[ct, st], [-st, ct]])

    # 旋转位置
    rotated_positions = np.einsum('ptc,ci->pti', xyv[:, :, :2], r)

    # 旋转速度
    rotated_velocities = np.einsum('ptc,ci->pti', xyv[:, :, 2:], r)

    # 合并旋转后的位置和速度
    rotated_xyv = np.concatenate((rotated_positions, rotated_velocities), axis=2)

    return rotated_xyv


def shift(xyv, center):
    # theta = random.random() * 2.0 * math.pi
    xyv = xyv - center[np.newaxis, np.newaxis, :]
    return xyv


def center_scene(xyv, obs_length=9, car_id=0, goals=None):
    if goals is not None:
        goals = goals[np.newaxis, :, :]
    # Center
    center = xyv[obs_length-1, car_id]  # Last Observation
    xyv = shift(xyv, center)
    if goals is not None:
        goals = shift(goals, center)
    # Rotate
    last_obs = xyv[obs_length-1, car_id]
    second_last_obs = xyv[obs_length-2, car_id]
    diff = np.array([last_obs[0] - second_last_obs[0], last_obs[1] - second_last_obs[1],
                        last_obs[2] - second_last_obs[2], last_obs[3] - second_last_obs[3]])
    theta = np.arctan2(diff[1], diff[0])
    rotation = -theta + np.pi/2
    xyv = theta_rotation(xyv, rotation)
    if goals is not None:
        goals = theta_rotation(goals, rotation)
        return xyv, rotation, center, goals[0]
    return xyv, rotation, center


def inverse_scene(xyv, rotation, center):
    xyv = theta_rotation(xyv, -rotation)
    xyv = shift(xyv, -center)
    return xyv


def drop_unobserved(xyv, obs_length=25):
    loc_at_obs = xyv[obs_length-1]
    absent_at_obs = np.isnan(loc_at_obs).any(axis=1)
    mask = ~absent_at_obs
    return xyv[:, mask], mask


def neigh_nan(xyv):
    return np.isnan(xyv).all()


def add_noise(observation, thresh=0.005, obs_length=25, ped='primary'):
    if ped == 'primary':
        observation[:obs_length, 0] += np.random.uniform(-thresh, thresh, observation[:obs_length, 0].shape)
    elif ped == 'neigh':
        observation[:obs_length, 1:] += np.random.uniform(-thresh, thresh, observation[:obs_length, 1:].shape)
    else:
        raise ValueError

    return observation
