import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.classifier import MnistClassifier
from tqdm import tqdm
import os
from dataset.mnist_dataset import MnistDataset
import yaml
import argparse
from utils.config_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_classifier():
    # Read the config file #
    with open(args.config_path, 'r') as file:  # open config file
        try:
            config = yaml.safe_load(file)  # YAML file -> python dictionary
        except yaml.YAMLError as exc:  # fail to open config file -> exception
            print(exc)
    print(config)
    ########################

    dataset_config = config['dataset_params']
    train_config = config['train_params']
    ldm_config = config['ldm_params']

    condition_config = get_config_value(ldm_config, key='condition_config', default_value=None)

    # Create the dataset #
    im_dataset_cls = {
        'mnist': MnistDataset,
        # 'adni': AdniDataset
    }.get(dataset_config['name'])

    train_dataset = im_dataset_cls(split='train',
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'],
                                use_latents=True,
                                latent_path=os.path.join(train_config['task_name'],
                                                         train_config['vae_latent_dir_name']),
                                condition_config=condition_config,
                                )

    # load data by batch unit
    # num_workers: the number of CPU threads
    train_loader = DataLoader(train_dataset,
                             batch_size=train_config['classifier_batch_size'],
                             shuffle=True,
                             num_workers=4)

    test_dataset = im_dataset_cls(split='test',
                               im_path=dataset_config['im_path'],
                               im_size=dataset_config['im_size'],
                               im_channels=dataset_config['im_channels'],
                               use_latents=True,
                               latent_path=os.path.join(train_config['task_name'],
                                                        train_config['vae_latent_dir_name']),
                               condition_config=condition_config,
                                )

    test_loader = DataLoader(test_dataset,
                              batch_size=train_config['classifier_batch_size'],
                              shuffle=True,
                              num_workers=4)

    # === 모델, 손실, 옵티마이저 ===
    model = MnistClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for data, cond_input in tqdm(train_loader, desc=f"Epoch {epoch}"):
            data = data.to(device)
            target = cond_input['class'].to(device)
            optimizer.zero_grad()
            output = model(data)  # logits (B,10)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

        avg_loss = train_loss / total
        acc = 100. * correct / total
        print(f"Train Epoch: {epoch} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

        # === 검증 ===
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, cond_input in test_loader:
                data = data.to(device)
                target = cond_input['class'].to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item() * data.size(0)
                _, predicted = output.max(1)
                correct += predicted.eq(target).sum().item()
        avg_test_loss = test_loss / len(test_loader.dataset)
        test_acc = 100. * correct / len(test_loader.dataset)
        print(f"Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    # === 모델 저장 ===
    os.makedirs(train_config['task_name'], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(train_config['task_name'], train_config['classifier_ckpt_name']))
    print("Saved trained classifier pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for classifier training')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    train_classifier()
