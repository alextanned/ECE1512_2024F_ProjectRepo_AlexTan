import datetime
import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
import wandb
import torchinfo

from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, get_time, epoch, DiffAugment, ParamDiffAug
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
import kornia as K
import torch.distributed as dist
import torch.cuda.comm
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch_flops import TorchFLOPsByFX
from torchvision import datasets, transforms

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Define the custom Dataset
def get_dataset(dataset, data_root, transform=None, partition='train', syn=False):

    if dataset == 'MHIST':
        if syn:
            syn_file = torch.load("syn/mhist/res_DataDAM_MHIST_ConvNetD7_50ipc_iter10.pt")
            syn_images = syn_file['data'][-1][0] # [100, 3, 224, 224]
            labels = torch.tensor([i for i in range(2) for _ in range(50)])

            dataset = TensorDataset(syn_images, labels)
            return dataset
        
        mean = [0.73943309, 0.65267006, 0.77742641]
        std = [0.19637706, 0.24239548, 0.16935587]
        images = [] 
        labels = []  
        annotations = pd.read_csv(os.path.join(data_root, 'annotations.csv'))
        for idx in range(len(annotations)):
            if annotations.iloc[idx, 3] == partition:
                img_name = os.path.join(data_root, 'images', annotations.iloc[idx, 0])
                image = np.array(Image.open(img_name).convert("RGB"), dtype=float)/255.0
                
                label = 1 if annotations.iloc[idx, 1] == "SSA" else 0
                image = (image - mean)/std
                image = np.transpose(image, (2, 0, 1))
                images.append(image)
                labels.append(label)
        images = np.array(images)
        labels = np.array(labels)

        dataset = TensorDataset(torch.tensor(images), torch.tensor(labels))

    
    elif dataset == 'MNIST':
        if syn:
            syn_file = torch.load("syn/mnist/res_DataDAM_MNIST_ConvNetD3_10ipc_iter10.pt")
            syn_images = syn_file['data'][-1][0] # [100, 1, 28, 28]
            labels = torch.tensor([i for i in range(10) for _ in range(10)])

            dataset = TensorDataset(syn_images, labels)
            return dataset
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        if partition == 'train':
            dataset = datasets.MNIST(data_root, train=True, download=True, transform=transform)
        elif partition == 'test':
            dataset = datasets.MNIST(data_root, train=False, download=True, transform=transform)

        
    
    return dataset

def define_architecture():
    """Define a random CNN architecture for NAS."""
    # Define the search space by choosing random parameters for the CNN
    np.random.seed(int(time.time() * 1000) % 100000)
    num_conv_layers = np.random.randint(1, 4) 
    conv_channels = [np.random.choice([16, 32, 64]) for _ in range(num_conv_layers)]
    kernel_sizes = [np.random.choice([3, 5]) for _ in range(num_conv_layers)]

    architecture = {
        'num_conv_layers': num_conv_layers,
        'conv_channels': conv_channels,
        'kernel_sizes': kernel_sizes,
    }
    return architecture

def build_model(architecture, channels=3, num_classes=2, im_size=(224, 224)):
    """Constructs a CNN model based on the chosen architecture."""
    layers = []

    # Add convolutional layers
    for out_channels, kernel_size in zip(architecture['conv_channels'], architecture['kernel_sizes']):
        layers.append(nn.Conv2d(channels, out_channels, kernel_size=kernel_size, padding='same'))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        channels = out_channels

    # Flatten and add fully connected layers
    layers.append(nn.Flatten())
    fc_input_dim = (im_size[0] // (2 ** architecture['num_conv_layers'])) ** 2 * channels 

    layers.append(nn.Linear(fc_input_dim, num_classes))  

    model = nn.Sequential(*layers)
    return model
    
def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--epochs', type=int, default=20000, help='training iterations')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--bs', type=int, default=64, help='batch size for training networks')   
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='', help='dataset path')
    parser.add_argument('--save_path', type=str, default='', help='path to save results')
    parser.add_argument('--flops', action='store_true',default=False,help='flops only mode')
    parser.add_argument('--syn', action='store_true',default=False,help='use synthesized image')
    parser.add_argument('--num_trials', type=int, default=5)
    
    args = parser.parse_args()
    args.method = 'Train'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if args.syn:
        train_dataset = get_dataset(args.dataset, args.data_path, partition='train', syn=True)
    else:
        train_dataset = get_dataset(args.dataset, args.data_path, partition='train')
    print(len(train_dataset))
    test_dataset = get_dataset(args.dataset, args.data_path, partition='test')

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=0)
    
    best_accuracy = 0
    best_architecture = None

    for trial in range(args.num_trials):

        architecture = define_architecture()


        if args.dataset == 'MHIST':
            model = build_model(architecture, channels=3, num_classes=2, im_size=(224, 224)).to(args.device)
        elif args.dataset == 'MNIST':
            model = build_model(architecture, channels=1, num_classes=10, im_size=(28, 28)).to(args.device)


        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*args.epochs)
        criterion = nn.CrossEntropyLoss().to(args.device)

        
        for i in range(args.epochs):

            print("")
            print('======== Epoch {:} / {:} ========'.format(i + 1, args.epochs))
            print('Training...')
            loss_avg, acc_avg, num_exp = 0, 0, 0

            model.train()
            t0 = time.time()

            for i_batch, datum in enumerate(train_loader):
                optimizer.zero_grad()
                img = datum[0].float().to(args.device)
                if args.dsa:
                    img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)

                lab = datum[1].to(args.device)
                bs = lab.shape[0]

                output = model(img)
                loss = criterion(output, lab)
                acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

                loss_avg += loss.item()*bs
                acc_avg += acc
                num_exp += bs
                if i_batch % 1 == 0:
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(i_batch, len(train_loader), format_time(time.time() - t0)))

                loss.backward()
                optimizer.step()
                scheduler.step()

            loss_avg /= num_exp
            acc_avg /= num_exp

            print("  average training loss: {0:.3f}".format(loss_avg))
            print("  accuracy: {0:.2f}".format(acc_avg))
            # validation
            print("")
            print("Running testing...")
            test_loss_avg, test_acc_avg, test_num_exp = 0, 0, 0

            
            model.eval()
            with torch.no_grad():
                for i_batch, datum in enumerate(test_loader):
                    img = datum[0].float().to(args.device)
                    lab = datum[1].long().to(args.device)
                    bs = lab.shape[0]

                    output = model(img)
                    loss = criterion(output, lab)
                    acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

                    test_loss_avg += loss.item()*bs
                    test_acc_avg += acc
                    test_num_exp += bs

                test_loss_avg /= test_num_exp
                test_acc_avg /= test_num_exp

            print("  average testing loss: {0:.3f}".format(test_loss_avg))
            print("  test accuracy: {0:.2f}".format(test_acc_avg))
            wandb.log({"lr": scheduler.optimizer.param_groups[0]['lr'], "test loss": test_loss_avg, "train loss": loss_avg, "test accuracy": test_acc_avg, "train accuracy": acc_avg})
            if test_acc_avg > best_accuracy:
                best_accuracy = test_acc_avg
                best_architecture = architecture
            # print(format_time(time.time() - t0))
            # exit()
        print(f"Trial {trial + 1}/{args.num_trials}")
    
    print(f"Best architecture: {best_architecture} with accuracy: {best_accuracy:.4f}")
    torch.save(best_architecture, os.path.join(args.save_path, "best_architecture.pt"))

if __name__ == '__main__':
    wandb.login()
    wandb.init(project="1512")
    main()

