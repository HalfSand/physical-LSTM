import torch
import os


def save_data(input_sequences, target_sequences, scale_factors, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(input_sequences, os.path.join(save_dir, 'input_sequences.pt'))
    torch.save(target_sequences, os.path.join(save_dir, 'target_sequences.pt'))
    torch.save(scale_factors, os.path.join(save_dir, 'scale_factors.pt'))

    print(f"Data saved in directory: {save_dir}")

def load_data(save_dir, device):
    input_sequences = torch.load(os.path.join(save_dir, 'input_sequences.pt')).to(device)
    target_sequences = torch.load(os.path.join(save_dir, 'target_sequences.pt')).to(device)
    scale_factors = torch.load(os.path.join(save_dir, 'scale_factors.pt')).to(device)

    print(f"Data loaded from directory: {save_dir}")
    return input_sequences, target_sequences, scale_factors


def save_model(epoch, model, optimizer, train_losses, val_losses, file_path='best_model.pth'):
    save_dict = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }
    torch.save(save_dict, file_path)
    print(f"Model saved in directory: {file_path}")


def load_model(model, optimizer=None, file_path='best_model.pth', return_epoch=False, return_losses=False):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    results = []
    if return_epoch:
        epoch = checkpoint.get('epoch', None)
        results.append(epoch)
    if return_losses:
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        results.extend([train_losses, val_losses])

    return results if len(results) > 0 else None