import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
from datasets.block import BlockDataset, LatentBlockDataset
import numpy as np


def load_cifar():
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train = datasets.CIFAR10(root="data", train=True, download=True, transform=t)
    val = datasets.CIFAR10(root="data", train=False, download=True, transform=t)
    return train, val, None  # CIFAR10 无单独 test


def load_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/randact_traj_length_100_n_trials_1000_n_contexts_1.npy'

    train = BlockDataset(data_file_path, train=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ]))

    val = BlockDataset(data_file_path, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(
                               (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]))
    return train, val, None

class FlatImageFolder(torch.utils.data.Dataset):
    """加载扁平目录下的图像（无类别子目录，如 ImageNet test）"""
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.files = sorted([
            f for f in os.listdir(root)
            if f.lower().endswith(('.jpeg', '.jpg', '.png', '.webp'))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        from PIL import Image
        path = os.path.join(self.root, self.files[idx])
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, -1  # 无标签


def load_imagenet(data_root):
    """
    Load ImageNet from folder structure: data_root/train/, data_root/val/, data_root/test/
    train/val: class subdirs; test: flat (optional)
    """
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    test_dir = os.path.join(data_root, 'test')
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f'ImageNet train dir not found: {train_dir}')
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f'ImageNet val dir not found: {val_dir}')

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train = datasets.ImageFolder(root=train_dir, transform=transform)
    # val: ILSVRC 可能为 class 子目录或扁平结构
    val_subdirs = [x for x in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, x)) and not x.startswith('.')]
    val_imgs = [f for f in os.listdir(val_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png', '.webp'))]
    if val_subdirs:
        val = datasets.ImageFolder(root=val_dir, transform=transform)
    elif val_imgs:
        val = FlatImageFolder(root=val_dir, transform=transform)
    else:
        raise FileNotFoundError(f'ImageNet val dir has no class subdirs or images: {val_dir}')
    test = None
    if os.path.isdir(test_dir) and len([f for f in os.listdir(test_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png', '.webp'))]) > 0:
        test = FlatImageFolder(root=test_dir, transform=transform)
    return train, val, test


def load_latent_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + '/data/latent_e_indices.npy'
    train = LatentBlockDataset(data_file_path, train=True, transform=None)
    val = LatentBlockDataset(data_file_path, train=False, transform=None)
    return train, val, None


def data_loaders(train_data, val_data, batch_size, test_data=None):
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size,
                            shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                             shuffle=False, pin_memory=True) if test_data else None
    return train_loader, val_loader, test_loader


def load_data_and_data_loaders(dataset, batch_size, data_root=None):
    if dataset == 'CIFAR10':
        training_data, validation_data, test_data = load_cifar()
        training_loader, validation_loader, test_loader = data_loaders(
            training_data, validation_data, batch_size, test_data)
        # torchvision 新版用 data，旧版用 train_data
        raw = getattr(training_data, 'data', getattr(training_data, 'train_data', None))
        x_train_var = np.var(raw / 255.0)

    elif dataset == 'BLOCK':
        training_data, validation_data, test_data = load_block()
        training_loader, validation_loader, test_loader = data_loaders(
            training_data, validation_data, batch_size, test_data)
        x_train_var = np.var(training_data.data / 255.0)

    elif dataset == 'IMAGENET':
        if not data_root:
            raise ValueError('--data_root required for ImageNet dataset')
        training_data, validation_data, test_data = load_imagenet(data_root)
        training_loader, validation_loader, test_loader = data_loaders(
            training_data, validation_data, batch_size, test_data)
        # 从样本估计方差（ImageFolder 无法整集加载）
        n_sample = min(1000, len(training_data))
        sample_loader = torch.utils.data.DataLoader(
            training_data, batch_size=batch_size, shuffle=True, num_workers=0)
        samples = []
        for x, _ in sample_loader:
            samples.append(x)
            if sum(s.shape[0] for s in samples) >= n_sample:
                break
        samples = torch.cat(samples, dim=0)[:n_sample]
        x_train_var = float(np.var(samples.numpy()))

    elif dataset == 'LATENT_BLOCK':
        training_data, validation_data, test_data = load_latent_block()
        training_loader, validation_loader, test_loader = data_loaders(
            training_data, validation_data, batch_size, test_data)
        x_train_var = np.var(training_data.data)

    else:
        raise ValueError(
            'Invalid dataset: only CIFAR10, BLOCK, IMAGENET are supported.')

    return training_data, validation_data, training_loader, validation_loader, test_loader, x_train_var


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def evaluate(model, data_loader, x_train_var, device):
    """在 val/test 上计算 recon_loss 和 perplexity"""
    model.eval()
    recon_losses, perplexities = [], []
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            embedding_loss, x_hat, perplexity = model(x)
            recon_loss = torch.mean((x_hat - x) ** 2) / x_train_var
            recon_losses.append(recon_loss.cpu().item())
            perplexities.append(perplexity.cpu().item())
    model.train()
    return np.mean(recon_losses), np.mean(perplexities)


def save_model_and_results(model, results, hyperparameters, timestamp, model_type='vqvae'):
    SAVE_MODEL_PATH = os.getcwd() + '/results'
    prefix = 'vae' if model_type == 'vae' else 'vqvae'

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save,
               SAVE_MODEL_PATH + '/' + prefix + '_data_' + timestamp + '.pth')
