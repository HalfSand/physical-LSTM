import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from .tools import load_model, load_data
import argparse
import numpy
from .modelA import MyModel

def losses_plot(train_losses, val_losses, save_path='losses.png'):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.show()


def traj_plot(scene_id, train_input, train_target, train_scale, model, criterion, save_path='traj.png'):
    model.zero_grad()
    model.eval()  # Set the model to evaluation mode
    model.set_scale_factors(train_scale)

    # Select the appropriate data from the given scene ID
    scene_data_input = train_input[scene_id]
    scene_data_target = train_target[scene_id]

    # Add a third dimension to match the expected input for the model
    scene_data_input = scene_data_input.unsqueeze(0)
    scene_data_target = scene_data_target.unsqueeze(0)

    # Compute the predictions
    with torch.no_grad():
        train_pred = model(scene_data_input, scene_data_target)

    # Calculate the loss
    train_loss = criterion(train_pred, scene_data_target[:, :, 2:])

    # Convert the loss to a percentage
    train_loss_percent = train_loss.item() * 100

    # Convert the data to numpy format for plotting
    scene_data_input_np = scene_data_input.squeeze().cpu().numpy()
    scene_data_target_np = scene_data_target.squeeze()[:, 2:4].cpu().numpy()
    train_pred_np = train_pred.squeeze().cpu().numpy()

    # Plot the known trajectory with thicker dashed line
    plt.plot(scene_data_input_np[:, 2], scene_data_input_np[:, 3], 'b--', linewidth=2, label='Traj_past')
    # Add small circles at each frame for the past trajectory
    plt.scatter(scene_data_input_np[:, 2], scene_data_input_np[:, 3], c='b', s=5)

    # Plot the predicted trajectory
    plt.plot(train_pred_np[:, 0], train_pred_np[:, 1], 'g-', label='Traj_pred')
    # Add small circles at each frame for the predicted trajectory
    plt.scatter(train_pred_np[:, 0], train_pred_np[:, 1], c='g', s=5)

    # Plot the actual trajectory
    plt.plot(scene_data_target_np[:, 0], scene_data_target_np[:, 1], 'r-', label='Traj_target')
    # Add small circles at each frame for the actual trajectory
    plt.scatter(scene_data_target_np[:, 0], scene_data_target_np[:, 1], c='r', s=5)

    # Mark the transition point with a triangle
    plt.scatter(scene_data_input_np[-1, 2], scene_data_input_np[-1, 3], marker='^', s=100, c='black',
                label='Current State')

    # Display the loss in the upper right corner of the plot
    plt.text(0.95, 0.01, f'Loss: {train_loss_percent:.2f}%',
             verticalalignment='bottom', horizontalalignment='right',
             transform=plt.gca().transAxes)

    # Add legend
    plt.legend()
    plt.savefig(save_path)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plotting functions")
    parser.add_argument("--plot_type", choices=['losses', 'trajectory'], default='losses',
                        help="Choose the type of plot (default: losses)")
    parser.add_argument("--model_path", default='best_model.pth', help="Path to the model (default: 'best_model.pth')")
    parser.add_argument("--scene_id", type=int, default=0, help="Scene ID for trajectory plot (default: 0)")
    parser.add_argument("--data_path", default= 'data_washed/test',
                        help="Path to your data (default:  'data_washed/test')")
    # 其他可能需要的参数
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 加载模型
    # hyper parameters
    input_size = 8  # frame, car_id, x, y, xVelocity, yVelocity, dx, dy, dvx, dvy
    hidden_size = 8  # LSTM hidden size
    num_layers = 1  # LSTM layers
    output_size = 2  # ax, ay as output

    model = MyModel(input_size, hidden_size, num_layers, output_size, device)
    model.to(device)

    # 定义损失函数
    criterion = nn.MSELoss()
    criterion.to(device)

    epoch, train_losses, val_losses = load_model(model, file_path=args.model_path, return_epoch=True, return_losses=True)
    # 加载数据
    test_input, test_target, test_scale = load_data(args.data_path, device)

    if args.plot_type == 'losses':
        losses_plot(train_losses, val_losses)
    elif args.plot_type == 'trajectory':
        traj_plot(args.scene_id, test_input, test_target, test_scale, model, criterion)

if __name__ == "__main__":
    main()





