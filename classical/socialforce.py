import numpy as np
from scipy.interpolate import interp1d
import torch

import socialforce
from socialforce.potentials import PedPedPotential
from socialforce.field_of_view import FieldOfView


def predict_sf(input_paths, dest_dict=None, dest_type='interp', sf_params=[0.5, 5.0, 0.3],
               predict_all=True, n_predict=25, obs_length=25):
    pred_length = n_predict

    def init_states(input_paths, start_frame, dest_dict, dest_type):
        initial_state = []
        for i, _ in enumerate(input_paths):
            path = input_paths[i]
            ped_id = path[0].car_id
            past_path = [t for t in path if t.frame <= start_frame]
            past_frames = [t.frame for t in path if t.frame <= start_frame]
            future_path = [t for t in path if t.frame > start_frame]
            len_path = len(past_path)

            ## To consider agent or not consider.
            if start_frame in past_frames:
                curr = past_path[-1]

                ## Velocity
                if len_path >= 4:
                    stride = 3
                    prev = past_path[-4]
                else:
                    stride = len_path - 1
                    prev = past_path[-len_path]
                [v_x, v_y] = vel_state(prev, curr, stride)

                ## Destination
                if dest_type == 'true':
                    if dest_dict is not None:
                        [d_x, d_y] = dest_dict[ped_id]
                    else:
                        raise ValueError
                elif dest_type == 'interp':
                    [d_x, d_y] = dest_state(past_path, len_path)
                elif dest_type == 'vel':
                    [d_x, d_y] = [pred_length * v_x, pred_length * v_y]
                elif dest_type == 'pred_end':
                    [d_x, d_y] = [future_path[-1].x, future_path[-1].y]
                else:
                    raise NotImplementedError

                ## Initialize State
                initial_state.append([curr.x, curr.y, v_x, v_y, d_x, d_y])
        return np.array(initial_state)

    def vel_state(prev, curr, stride):
        if stride == 0:
            return [0, 0]
        diff = np.array([curr.x - prev.x, curr.y - prev.y])
        theta = np.arctan2(diff[1], diff[0])
        speed = np.linalg.norm(diff) / (stride * 0.4)
        return [speed * np.cos(theta), speed * np.sin(theta)]

    def dest_state(path, length):
        if length == 1:
            return [path[-1].x, path[-1].y]
        x = [t.x for t in path]
        y = [t.y for t in path]
        time = list(range(length))
        f = interp1d(x=time, y=[x, y], fill_value='extrapolate')
        return f(time[-1] + pred_length)

    multimodal_outputs = {}
    primary = input_paths[0]
    neighbours_tracks = []
    frame_diff = primary[1].frame - primary[0].frame
    start_frame = primary[obs_length - 1].frame
    first_frame = primary[obs_length - 1].frame + frame_diff

    # initialize
    initial_state = init_states(input_paths, start_frame, dest_dict, dest_type)

    fps = 25
    sampling_rate = int(fps / 1)

    #     if len(initial_state) != 0:
    #         # run
    #         ped_ped = PedPedPotential(v0=sf_params[1], sigma=sf_params[2])
    #         field_of_view = FieldOfView()
    #         s = socialforce.Simulator(ped_ped=ped_ped, field_of_view=field_of_view,
    #                                   delta_t=1./fps, tau=sf_params[0])
    #         states = np.stack([s.step().state.copy() for _ in range(pred_length*sampling_rate)])
    #         ## states : pred_length x num_ped x 7
    #         states = np.array([s for num, s in enumerate(states) if num % sampling_rate == 0])

    if len(initial_state) != 0:
        # 准备参数和模拟器
        ped_ped = PedPedPotential(v0=sf_params[1], sigma=sf_params[2])
        field_of_view = FieldOfView()
        s = socialforce.Simulator(ped_ped=ped_ped, field_of_view=field_of_view,
                                  delta_t=1. / fps, tau=sf_params[0])

        # 确保initial_state是正确的数据类型和格式
        initial_state = torch.tensor(initial_state, dtype=torch.float32)  # 以您的实际initial_state来修改

        # 执行模拟
        number_of_steps = pred_length * sampling_rate  # 根据您的需求修改
        result_states = s.run(initial_state, number_of_steps)

        # 如果您需要的states是numpy array类型，您可以这样转换：
        states = result_states.detach().numpy()

        # 您可以根据需要处理和过滤模拟的结果
        states = np.array([s for num, s in enumerate(states) if num % sampling_rate == 0])

    else:
        ## Stationary
        past_path = [t for t in input_paths[0] if t.frame == start_frame]
        states = np.stack([[[past_path[0].x, past_path[0].y]] for _ in range(pred_length)])

    # predictions
    primary_track = states[:, 0, 0:2]
    neighbours_tracks = states[:, 1:, 0:2]

    ## Primary Prediction Only
    if not predict_all:
        neighbours_tracks = []

    # Unimodal Prediction
    multimodal_outputs[0] = primary_track, neighbours_tracks
    return multimodal_outputs

