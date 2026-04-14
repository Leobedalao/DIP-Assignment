import os
import csv
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from FCN_network import FullyConvNetwork
from torch.optim.lr_scheduler import StepLR

run_name = 'cityscapes'  # 'facades' or 'cityscapes'
train_list_file = 'cityscapes_train_list.txt'
val_list_file = 'cityscapes_val_list.txt'

output_root = os.path.join('outputs', run_name)
train_result_dir = os.path.join(output_root, 'train_results')
val_result_dir = os.path.join(output_root, 'val_results')
checkpoint_dir = os.path.join(output_root, 'checkpoints')
log_dir = os.path.join(output_root, 'training_logs')


def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image


def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images.
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    epoch_dir = os.path.join(folder_name, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)

    num_images = min(num_images, inputs.shape[0])
    for i in range(num_images):
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        comparison = np.hstack((input_img_np, target_img_np, output_img_np))
        cv2.imwrite(os.path.join(epoch_dir, f'result_{i + 1}.png'), comparison)


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, num_epochs):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer for updating model parameters.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the training on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    model.train()
    running_loss = 0.0

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        optimizer.zero_grad()
        outputs = model(image_rgb)

        if epoch % 5 == 0 and i == 0:
            save_images(image_rgb, image_semantic, outputs, train_result_dir, epoch)

        loss = criterion(outputs, image_semantic)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

    avg_train_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}')
    return avg_train_loss


def validate(model, dataloader, criterion, device, epoch, num_epochs):
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            outputs = model(image_rgb)
            loss = criterion(outputs, image_semantic)
            val_loss += loss.item()

            if epoch % 5 == 0 and i == 0:
                save_images(image_rgb, image_semantic, outputs, val_result_dir, epoch)

    avg_val_loss = val_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')
    return avg_val_loss


def save_loss_table(history, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'train_loss', 'val_loss'])
        for row in history:
            writer.writerow([row['epoch'], row['train_loss'], row['val_loss']])


def save_loss_curve(history, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    epochs = [row['epoch'] for row in history]
    train_losses = [row['train_loss'] for row in history]
    val_losses = [row['val_loss'] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    """
    Main function to set up the training and validation processes.
    """
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    train_dataset = FacadesDataset(list_file=train_list_file)
    val_dataset = FacadesDataset(list_file=val_list_file)

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=4)

    model = FullyConvNetwork().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))

    scheduler = StepLR(optimizer, step_size=200, gamma=0.2)
    history = []

    num_epochs = 300
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs)
        val_loss = validate(model, val_loader, criterion, device, epoch, num_epochs)
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
        })

        scheduler.step()

        if (epoch + 1) % 50 == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'pix2pix_model_epoch_{epoch + 1}.pth'))

    save_loss_table(history, os.path.join(log_dir, 'loss_history.csv'))
    save_loss_curve(history, os.path.join(log_dir, 'loss_curve.png'))


if __name__ == '__main__':
    main()
