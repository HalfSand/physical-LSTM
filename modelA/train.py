import torch
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
import argparse
import matplotlib.pyplot as plt
from .tools import load_data, save_model, load_model


class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(MyModel, self).__init__()
        self.l1 = nn.Linear(input_size, output_size)  # 线性层
        self.tanh = nn.Tanh()
        self.l2 = nn.Linear(output_size, output_size)
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers)  # LSTM层
        self.reset_parameters()
        self.scale_factors = None
        self.device = device

    def set_scale_factors(self, scale_factors):
        self.scale_factors = scale_factors
#必须加一层循环，更新下一个frame的整个map，才能用预测来预测
    def forward(self, input_sequences, target_sequences):
        train_pred = torch.zeros_like(target_sequences[:, :, 2:])  # x, y ,xV, yV, relatives x,y... 8
        index = 0

        for input_seq in input_sequences:
            # pooling operation
            input_seq = self.pooling_operation(input_seq, input_sequences)
            target_seq = target_sequences[index]
            frames = target_seq[:, 0]
            car_id = target_seq[-1, 1]
            pred_pre = torch.empty(len(frames), 6)

            for i in range(len(frames)):
                # 输入序列经过线性层计算fx和fy
                fx_fy = self.l1(input_seq)
                fx_fy = self.tanh(fx_fy)
                fx_fy = self.l2(fx_fy)

                # LSTM接收fx和fy序列作为输入
                _, (acc, _) = self.lstm(fx_fy)  # acc : 1*2: frame: 11
                acc = acc.squeeze()  # acceleration as latent variable
                ######################################################################\\\\ Use prediction to do prediction\\\
                # 计算预测值 Decoder
                #Here we must scale the data back, because our model is with a physical meaning.
                input_seq = input_seq.clone()
                input_seq[:, :4] = input_seq[:, :4] * self.scale_factors

                x = input_seq[-1, 0]  # x for the last frame
                y = input_seq[-1, 1]  # y for the last frame
                x_velocity = input_seq[-1, 2]  # xVel for the last frame
                y_velocity = input_seq[-1, 3]  # yVel for the last frame

                dt = 0.04  # fps is 25, so dt = 1 / 25 = 0.04s
                x_velocity_pred = x_velocity + dt * acc[0]
                y_velocity_pred = y_velocity + dt * acc[1]
                x_pred = x + dt * x_velocity_pred
                y_pred = y + dt * y_velocity_pred
                # 想办法做一个pooling操作，加到input_seq后面

                pred_unscaled = torch.stack([frames[i], car_id, x_pred, y_pred, x_velocity_pred, y_velocity_pred], dim=0).to(self.device)
                pred_unscaled_temp = pred_unscaled.clone()
                pred_unscaled_temp[2:] = pred_unscaled[2:] / self.scale_factors
                pred_unscaled = pred_unscaled_temp
                pred_pre[i] = pred_unscaled
                pooled = self.pooling_operation(pred_pre[i].unsqueeze(0), input_sequences)
                input_seq = torch.cat((input_seq, pooled), dim=0)
                input_seq = input_seq[1:, :]  # 删除第一行

            train_pred[index] = pred_pre[:, 2:]
            index = index + 1
        return train_pred

    def reset_parameters(self):
        # Reset parameters for linear and LSTM layers
        self.l1.reset_parameters()
        self.l2.reset_parameters()
        self.lstm.reset_parameters()

    def pooling_operation(self, input_seq, input_sequences):
        frames = input_seq[:, 0].long()
        car_ids = input_seq[:, 1].long()[0]  # 提取car_id

        # 初始化空的Tensor列表，并将它们放在GPU上
        pooled_outputs = [torch.empty(0, 8, device=self.device) for _ in range(len(frames))]
        pooled_pre = [torch.empty(0, 8, device=self.device) for _ in range(len(frames))]

        for i in range(len(frames)):
            mask = torch.nonzero(input_sequences[:, :, 0].long() == frames[i])  # 寻找同时出现的车辆的索引
            m = mask[:, 0]
            n = mask[:, 1]

            # 计算差值的累加和
            dx = torch.sum(input_seq[i, 2] - input_sequences[m, n, 2])
            dy = torch.sum(input_seq[i, 3] - input_sequences[m, n, 3])
            dvx = torch.sum(input_seq[i, 4] - input_sequences[m, n, 4])
            dvy = torch.sum(input_seq[i, 5] - input_sequences[m, n, 5])

            # 将差值累加和组合成一个张量
            pooled_output = torch.tensor(
                [input_seq[i, 2], input_seq[i, 3], input_seq[i, 4], input_seq[i, 5], dx, dy, dvx, dvy])
            pooled_pre[i] = pooled_output
        pooled_outputs = torch.cat(pooled_pre).reshape(len(frames), 8).float().to(self.device)
        return pooled_outputs



# 训练模型
def train_model(model, train_input, train_target, train_scale, val_input, val_target, val_scale,
                optimizer, criterion, num_epochs=50, patience=5, start_epoch = 0):

    train_losses = []
    val_losses = []
    writer = SummaryWriter('logs')
    model.zero_grad()
    model.train()  # 设置模型为训练模式
    best_loss = float('inf')  # 初始化最佳损失为正无穷大
    early_stop_count = 0

    for epoch in range(start_epoch, num_epochs):
        model.set_scale_factors(train_scale)
        optimizer.zero_grad()  # 清零梯度

        # 前向传播
        train_pred = model(train_input, train_target)

        # 计算损失
        train_loss = criterion(train_pred[:, :, 0:2], train_target[:, :, 2:4])
        train_losses.append(train_loss.item())

        # 反向传播和优化
        train_loss.backward()
        optimizer.step()

        model.eval()  # 设置模型为评估模式
        model.set_scale_factors(val_scale)

        with torch.no_grad():
            val_pred = model(val_input, val_target)
            val_loss = criterion(val_pred[:, :, 0:2], val_target[:, :, 2:4])
            val_losses.append(val_loss.item())

        if val_loss < best_loss:
            best_loss = val_loss
            save_model(epoch, model, optimizer, train_losses, val_losses, file_path='best_model.pth')
            early_stop_count = 0  # 重置early stop计数
        else:
            early_stop_count += 1  # 没有改善的次数加1

        if early_stop_count >= patience:
            print("Early stopping.")
            break  # 如果连续patience次没有改善，就停止训练

        model.train()  # 恢复模型为训练模式

        loss_percentage_train = train_loss.item() * 100
        writer.add_scalar('Loss/train', loss_percentage_train, epoch)
        loss_percentage_val = val_loss.item() * 100
        writer.add_scalar('Loss/validation', loss_percentage_val, epoch)

        print(f'Epoch [{epoch + 1}/{num_epochs}]:\nTraining Loss: {loss_percentage_train:2f}%')
        print(f'Validation Loss: {loss_percentage_val:2f}%')
        print(f'train_pred:\n{train_pred[1] * train_scale}')
        print(f'train_target:\n{train_target[1, :, 2:] * train_scale}')
        print(f'val_pred:\n{val_pred[1] * val_scale}')
        print(f'val_target:\n{val_target[1, :, 2:] * val_scale}')

    writer.close()


def test_model(model, test_input, test_target, test_scale, criterion):
    model.zero_grad()
    model.eval()  # 设置模型为评估模式
    model.set_scale_factors(test_scale)

    with torch.no_grad():
        test_pred = model(test_input, test_target)
        test_loss = criterion(test_pred[:, :, 0:2], test_target[:, :, 2:4])
        loss_percentage_test = test_loss.item() * 100
        print(f'Test Loss: {loss_percentage_test:2f}%')
        print(f'test_pred:\n{test_pred[1] * test_scale}')
        print(f'test_target:\n{test_target[1, :, 2:] * test_scale}')




def parse_args():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--resume_training', type=bool, default=False, help='Resume training from saved model')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    resume_training = args.resume_training

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)

    # read data
    file_path1 = 'data_washed/train'
    file_path2 = 'data_washed/val'
    file_path3 = 'data_washed/test'
    train_input, train_target, train_scale = load_data(file_path1, device)
    val_input, val_target, val_scale = load_data(file_path2, device)
    test_input, test_target, test_scale = load_data(file_path3, device)

    # hyper parameters
    input_size = 8  # frame, car_id, x, y, xVelocity, yVelocity, dx, dy, dvx, dvy
    hidden_size = 8  # LSTM hidden size
    num_layers = 1  # LSTM layers
    output_size = 2  # ax, ay as output

    # 定义模型
    model = MyModel(input_size, hidden_size, num_layers, output_size, device)
    model.to(device)

    # 定义损失函数
    criterion = nn.MSELoss()
    criterion.to(device)

    # 定义优化器
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 设置训练的超参数
    num_epochs = 100
    patience = 5

    # 检查是否存在保存的模型权重
    start_epoch = 0
    saved_model_path = 'best_model.pth'
    if os.path.exists(saved_model_path) and resume_training:
        print("Loading previously saved model...")
        start_epoch = load_model(model, optimizer, file_path=saved_model_path, return_epoch=True)

    # 开始训练
    train_model(model, train_input.float(), train_target.float(), train_scale.float(), val_input.float(), val_target.float(),
                val_scale.float(), optimizer, criterion, num_epochs, patience, start_epoch)

    # 加载保存的最佳模型
    best_model = MyModel(input_size, hidden_size, num_layers, output_size, device).to(device)
    load_model(best_model, file_path=saved_model_path)

    # 在test_set上进行测试
    test_model(best_model, test_input.float(), test_target.float(), test_scale.float(), criterion)




if __name__ == '__main__':
    main()
