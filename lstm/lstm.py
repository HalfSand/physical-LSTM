import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from trajnetplusplustools import Reader
from .utils import center_scene, inverse_scene


class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device, scale_factors=None):
        super(MyModel, self).__init__()
        self.l1 = nn.Linear(input_size, output_size)  # Linear layer
        self.tanh = nn.Tanh()
        self.l2 = nn.Linear(output_size, output_size)
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers)  # LSTM layer
        self.reset_parameters()
        self.scale_factors = scale_factors \
            if scale_factors is not None else \
            torch.tensor([2, 0.5, 0.01, 0.02]).to(device) # scaler for ind
        self.device = device

    # torch.tensor([4.0, 0.3, 0.5, 0.015]).to(device) #scaler for highd

    def reset_parameters(self):
        # Reset parameters for linear and LSTM layers
        self.l1.reset_parameters()
        self.l2.reset_parameters()
        self.lstm.reset_parameters()

    def encoder(self, input_seq, lengths):
        fx_fy = self.l1(input_seq)
        fx_fy = self.tanh(fx_fy)
        fx_fy = self.l2(fx_fy)

        # Sort the sequences by length in descending order
        lengths, indices = lengths.sort(descending=True)
        fx_fy = fx_fy[:, indices]

        # Pack the sequences
        fx_fy_packed = pack_padded_sequence(fx_fy, lengths.cpu(), enforce_sorted=False)

        # Get the outputs from the LSTM
        # _, (acc, _) = self.lstm(fx_fy_packed)
        output, _ = self.lstm(fx_fy_packed)
        output_padded, _ = pad_packed_sequence(output, batch_first=True)
        acc = output_padded[:, -1]

        # Reshape the output
        acc = acc.squeeze(0)

        # Restore the original order
        _, inv_indices = indices.sort()
        acc = acc[inv_indices]

        return acc

    def decoder(self, last_positions, acc_tensor):
        fps = 25
        dt = 1 / fps  # Time step

        # Extract x, y, x_velocity, y_velocity from the last positions
        x, y, x_velocity, y_velocity = torch.chunk(last_positions, 4, dim=-1)

        # Compute predicted velocities
        x_velocity_pred = x_velocity + dt * acc_tensor[:, 0].unsqueeze(1)
        y_velocity_pred = y_velocity + dt * acc_tensor[:, 1].unsqueeze(1)

        # Compute predicted positions
        x_pred = x + dt * x_velocity_pred
        y_pred = y + dt * y_velocity_pred

        # Combine the predicted positions and velocities
        pred_frame = torch.cat([x_pred, y_pred, x_velocity_pred, y_velocity_pred], dim=-1)

        return pred_frame

    def forward(self, observed, prediction_truth, batch_split):
        # Unscale the observed data
        unscaled_observed = observed * self.scale_factors.unsqueeze(0).unsqueeze(0)

        # get all nan lines indices
        mask = torch.isnan(unscaled_observed[:, :, 0])
        nan_rows = mask.all(dim=0)
        indices_to_keep = torch.where(~nan_rows)[0]

        # Delete the all nan lines
        unscaled_observed = torch.index_select(unscaled_observed, 1, indices_to_keep)
        prediction_truth = torch.index_select(prediction_truth, 1, indices_to_keep)

        # Update the batch_split
        nan_indices = torch.nonzero(nan_rows).squeeze()
        for i in range(1, len(batch_split)):
            subtract_count = (nan_indices < batch_split[i]).sum().item()
            batch_split[i] -= subtract_count

        # Mask the observed data
        mask = torch.isnan(unscaled_observed[:, :, 0])
        unscaled_observed = torch.where(torch.isnan(unscaled_observed), torch.zeros_like(unscaled_observed),
                                        unscaled_observed)
        lengths = (~mask).sum(dim=0).cpu().long()
        pooled_positions = self.generate_pooling(unscaled_observed, batch_split)

        # Get each batch's primary_car for truth without loop
        primary_indices = batch_split[:-1]
        target_truth = prediction_truth[:, primary_indices].to(self.device)
        target_pred = torch.zeros_like(target_truth).to(self.device)
        pred_frames = target_truth.shape[0]

        for i in range(pred_frames):
            last_positions = pooled_positions[-1, :, :4]

            acc_tensor = self.encoder(pooled_positions, lengths)

            # Decode the acc_tensor to get the predicted positions and velocities for the current frame
            pred_frame = self.decoder(last_positions, acc_tensor)

            # Scale the predictions
            scaled_prediction = pred_frame / self.scale_factors.unsqueeze(0)

            # Store the prediction 应该是此frame对应的batch数量行，因为是primary cars。列数是4.
            target_pred[i] = scaled_prediction[primary_indices]

            # Generate pooling for the predicted frame
            pooled_pred_frame = self.generate_pooling(pred_frame, batch_split)

            # Add the new pooled_pred_frame to the end of pooled_positions and remove the first frame
            pooled_positions = torch.cat([pooled_positions[1:], pooled_pred_frame.unsqueeze(0)], dim=0)

        return target_pred, target_truth


    def generate_pooling(self, obs_data, batch_split):
        single_frame = False

        # Check if it's a single frame data
        if len(obs_data.shape) == 2:
            single_frame = True
            obs_data = obs_data.unsqueeze(0)  # Add a frame dimension

        num_frames, num_cars, _ = obs_data.shape
        curr_positions = obs_data.clone()  # Clone obs_data to get curr_positions

        # Create a mask where cars with all zeros are considered missing
        valid_cars = (obs_data.sum(dim=2) != 0).unsqueeze(2)

        # Expand dimensions to allow broadcasting
        expanded_obs = obs_data.unsqueeze(2)
        expanded_valid_cars = valid_cars.unsqueeze(2)

        # Calculate the differences using broadcasting
        diffs = expanded_obs - obs_data.unsqueeze(1)

        eps = 1e-10
        diffs[diffs == 0] += eps  # Add a very small value to zero entries

        # Calculate the inverse
        diffs_inv = 1.0 / diffs

        # Set values that are extremely large (because of the division by a small value) to zero
        threshold = 1e9  # You can adjust this threshold as needed
        diffs_inv[diffs_inv.abs() > threshold] = 0

        # Use the valid cars mask to ignore calculations involving missing cars
        diffs_inv *= expanded_valid_cars

        # Sum the inversed differences along the car axis
        scores = diffs_inv.sum(dim=2)

        # Concatenate along the third dimension
        pooled_positions = torch.cat((curr_positions, scores), dim=2)

        # Remove the added frame dimension if it's a single frame data
        if single_frame:
            pooled_positions = pooled_positions.squeeze(0)

        return pooled_positions


class LSTMPredictor(object):
    def __init__(self, model):
        self.model = model

    def save(self, state, filename):
        with open(filename, 'wb') as f:
            torch.save(self, f)

        # # during development, good for compatibility across API changes:
        # # Save state for optimizer to continue training in future
        with open(filename + '.state', 'wb') as f:
            torch.save(state, f)

    @staticmethod
    def load(filename, device):
        with open(filename, 'rb') as f:
            return torch.load(f, map_location=device)

    def __call__(self, paths, scene_goal, n_predict=12, modes=1, predict_all=True, obs_length=25, start_length=0,
                 args=None):
        self.model.eval()

        # self.model.train()
        with torch.no_grad():
            scene = Reader.paths_to_xyv(paths)
            # xyv = add_noise(xyv, thresh=args.thresh, ped=args.ped_type)
            batch_split = [0, scene.shape[1]]

            if args.normalize_scene:
                scene, rotation, center, scene_goal = center_scene(scene, obs_length, goals=scene_goal)

            scene = torch.Tensor(scene).to(args.device)
            # scene_goal = torch.Tensor(scene_goal)  # .to(device)
            batch_split = torch.Tensor(batch_split).long().to(args.device)

            multimodal_outputs = {}
            for num_p in range(modes):
                # _, output_scenes = self.model(xyv[start_length:obs_length], scene_goal, batch_split, xyv[obs_length:-1].clone())
                # _, output_scenes = self.model(xyv[start_length:obs_length], scene_goal, batch_split, n_predict=n_predict)

                observed = scene[start_length:obs_length].clone()
                # prediction_truth = scene[obs_length:].clone()
                prediction_truth = scene[obs_length:].clone()


                # Create a virtual batch
                num_copies = 3
                observed_batch = observed.repeat(1, num_copies, 1)
                prediction_truth_batch = prediction_truth.repeat(1, num_copies, 1)

                # Compute the original batch_split

                batch_split = batch_split.repeat(num_copies)




                output_scenes, _ = self.model(observed_batch, observed_batch, batch_split)
                output_scenes = output_scenes[:, 1]

                output_scenes = output_scenes.to('cpu').numpy()
                if args.normalize_scene:
                    output_scenes = inverse_scene(output_scenes, rotation, center)
                output_primary = output_scenes[:, :2]
                # if not predict_all:
                #     output_neighs = []
                # else:
                #     output_neighs = output_scenes[:, 1:]
                ## Dictionary of predictions. Each key corresponds to one mode
                multimodal_outputs[num_p] = [output_primary, []]

        ## Return Dictionary of predictions. Each key corresponds to one mode
        return multimodal_outputs
