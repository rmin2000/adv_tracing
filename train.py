import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os
import argparse

from models import VGG16Head, VGG16Tail, ResNet18Head, ResNet18Tail
import config
from watermark import Watermark


'''
Train the multi-head-one-tail model.
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help = 'Benchmark model structure.', choices = ['VGG16', 'ResNet18'])
    parser.add_argument('--dataset_name', help = 'Benchmark dataset used.', choices = ['CIFAR10', 'GTSRB', 'TINY'])
    parser.add_argument('--num_workers', help = 'Number of workers', type = int, default = 2)
    parser.add_argument('-N', '--num_heads', help = 'Number of heads.', type = int, default = 100)
    parser.add_argument('-b', '--batch_size', help = 'Batch size.', type = int, default = 128)
    parser.add_argument('-e', '--num_epochs', help = 'Number of epochs.', type = int, default = 10)
    parser.add_argument('-lr', '--learning_rate', help = 'Learning rate.', type = float, default = 1e-3)
    parser.add_argument('-md', '--masked_dims', help = 'Number of masked dimensions', type = int, default = 100)
    
    args = parser.parse_args()
    
    if args.dataset_name == 'CIFAR10' or args.dataset_name == 'GTSRB':
        C, H, W = 3, 32, 32
    elif args.dataset_name == 'tiny':
        C, H, W = 3, 64, 64

    # Create the model and the dataset
    dataset = eval(f'config.{args.dataset_name}()')
    training_set, testing_set = dataset.training_set, dataset.testing_set
    num_classes = dataset.num_classes
    means, stds = dataset.means, dataset.stds
    Head, Tail = eval(f'{args.model_name}Head'), eval(f'{args.model_name}Tail')
    normalizer = transforms.Normalize(means, stds)
    training_loader = torch.utils.data.DataLoader(training_set, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    testing_loader = torch.utils.data.DataLoader(testing_set, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Place to save the trained model
    save_dir = f'saved_models/{args.model_name}-{args.dataset_name}'
    os.makedirs(save_dir, exist_ok = True)

    # Load the tail of the model
    tail = Tail(num_classes)
    tail.load_state_dict(torch.load(f'{save_dir}/base_tail_state_dict'))
    
    tail.to(device)

    
    # training
    
    for i in range(args.num_heads):
        
        os.makedirs(f'{save_dir}/head_{i}', exist_ok = True)
        
        head = nn.Sequential(Watermark.random(args.masked_dims, C, H, W), Head())
        
        head.to(device)
        head[0].save(f'{save_dir}/head_{i}/watermark.npy')
        head[1].load_state_dict(torch.load(f'{save_dir}/base_head_state_dict'))
        optimizer = torch.optim.Adam(head.parameters(), lr = args.learning_rate)
        Loss = nn.CrossEntropyLoss()
        best_accuracy = 0.

        for n in range(args.num_epochs):
            head.train()
            epoch_mask_grad_norm, epoch_mask_grad_norm_inverse = 0., 0.
            epoch_loss = 0.0
            for X, y in training_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                out_clean = tail(head(normalizer(X)))
                clean_loss = Loss(out_clean, y)
                loss = clean_loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(y) / len(training_set)

            # testing
            head.eval()
            tail.eval()
            
            accuracy = 0.0
            with torch.no_grad():
                for X, y in testing_loader:
                    X, y = X.to(device), y.to(device)
                    _, pred = tail(head(normalizer(X))).max(axis = -1)
                    accuracy += (pred == y).sum().item() / len(testing_set)

            print(f'Head {i}, epoch {n}, loss {epoch_loss:.3f}, accuracy = {accuracy:.4f}')

            # save the best result
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(head[1].state_dict(), f'{save_dir}/head_{i}/state_dict')

        print(f'Completed the training for head {i}, accuracy = {best_accuracy:.4f}.')
    print(f'Completed the training of {args.num_heads} heads, {args.model_name}-{args.dataset_name}.')
