import torch
import pandas as pd
import os
from .tools import save_data

class FactorScaler:
    def __init__(self, device):
        self.scale_factors = None
        self.device = device
    def fit_transform(self, data):
        # self.scale_factors, _ = torch.abs(data).max(dim=0)
        self.scale_factors = torch.tensor([1, 1, 1, 1]).to(self.device)
        return data / self.scale_factors

    def transform(self, data):
        return data / self.scale_factors

    def inverse_transform(self, scaled_data):
        return scaled_data * self.scale_factors


class DataPreparer:
    def __init__(self, window_size, prediction_steps, device):
        self.window_size = window_size
        self.prediction_steps = prediction_steps
        self.scaler = FactorScaler(device)
        self.device = device

    def prepare_data(self, data_path, n):
        # Read data
        device = self.device
        data = pd.read_csv(data_path, delimiter='\t', header=None,
                           names=['frame', 'car_id', 'x', 'y', 'xVelocity', 'yVelocity'])[0:3 * n]

        # Data preprocessing
        grouped_data = data.groupby('car_id')

        # Create input features and target sequences
        input_sequences = []
        target_sequences = []

        for car_id, group in grouped_data:
            # Sort each car's trajectory data by frame
            sorted_group = group.sort_values('frame')

            # Extract features and target sequences
            for i in range(len(sorted_group) - self.window_size - self.prediction_steps + 1):
                input_seq = sorted_group[['frame', 'car_id', 'x', 'y', 'xVelocity', 'yVelocity']].values[
                            i:i + self.window_size]
                target_seq = sorted_group[['frame', 'car_id', 'x', 'y', 'xVelocity', 'yVelocity']].values[
                             i + self.window_size:i + self.window_size + self.prediction_steps]
                input_sequences.append(torch.tensor(input_seq, dtype=torch.float))
                target_sequences.append(torch.tensor(target_seq, dtype=torch.float))

                # Convert to PyTorch tensors
        input_sequences = torch.stack(input_sequences)[0:n].to(device)
        target_sequences = torch.stack(target_sequences)[0:n].to(device)

        # Scale input_sequences
        scaled_input_data = self.scaler.fit_transform(input_sequences[:, :, 2:].reshape(-1, 4))
        scaled_input_sequences = input_sequences.clone()
        scaled_input_sequences[:, :, 2:] = scaled_input_data.reshape(input_sequences[:, :, 2:].shape)

        # Scale target_sequences
        scaled_target_data = self.scaler.transform(target_sequences[:, :, 2:].reshape(-1, 4))
        scaled_target_sequences = target_sequences.clone()
        scaled_target_sequences[:, :, 2:] = scaled_target_data.reshape(target_sequences[:, :, 2:].shape)

        return scaled_input_sequences, scaled_target_sequences




def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)

    prediction_steps = 10
    window_size = 10

    prep_train = DataPreparer(window_size, prediction_steps, device)
    train_data = 'highD/train/01_tracks.txt'
    prep_val = DataPreparer(window_size, prediction_steps, device)
    val_data = 'highD/val/10_tracks.txt'
    prep_test = DataPreparer(window_size, prediction_steps, device)
    test_data = 'highD/test/09_tracks.txt'
    train_input, train_target = prep_train.prepare_data(train_data, 10000)
    train_scale = prep_train.scaler.scale_factors
    val_input, val_target = prep_val.prepare_data(val_data, 3000)
    val_scale = prep_val.scaler.scale_factors
    test_input, test_target = prep_test.prepare_data(test_data, 1000)
    test_scale = prep_test.scaler.scale_factors

    # 打印示例数据
    print("input_sequences examples:")
    print(test_input[0:5])
    print("target_sequences examples：")
    print(test_target[0:5])
    print(prep_train.scaler.scale_factors)

    # save data
    save_dir1 = 'data_washed/train'
    save_data(train_input, train_target, train_scale, save_dir1)
    save_dir2 = 'data_washed/val'
    save_data(val_input, val_target, val_scale, save_dir2)
    save_dir3 = 'data_washed/test'
    save_data(test_input, test_target, test_scale, save_dir3)

if __name__ == '__main__':
    main()
