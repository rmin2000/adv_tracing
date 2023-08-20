import torch
import torch.nn as nn
import os
import argparse

from torchvision import transforms
from models import VGG16Head, VGG16Tail, ResNet18Head, ResNet18Tail
import config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help = 'Benchmark model structure.', choices = ['VGG16', 'ResNet18'])
    parser.add_argument('--dataset_name', help = 'Benchmark dataset used.', choices = ['CIFAR10', 'GTSRB', 'TINY'])
    parser.add_argument('--num_workers', help = 'Number of workers', type = int, default = 2)
    parser.add_argument('-b', '--batch_size', help = 'Batch size.', type = int, default = 128)
    parser.add_argument('-e', '--num_epochs', help = 'Number of epochs.', type = int, default = 50)
    parser.add_argument('-lr', '--learning_rate', help = 'Learning rate.', type = float, default = 1e-3)
    args = parser.parse_args()

    # Create the model and the dataset
    dataset = eval(f'config.{args.dataset_name}()')
    training_set, testing_set = dataset.training_set, dataset.testing_set
    num_classes = dataset.num_classes
    means, stds = dataset.means, dataset.stds
    Head, Tail = eval(f'{args.model_name}Head'), eval(f'{args.model_name}Tail')
    base_model = nn.Sequential(transforms.Normalize(means, stds), Head(), Tail(num_classes))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_model.to(device)
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle = True, num_workers = args.num_workers)
    testing_loader = torch.utils.data.DataLoader(testing_set, batch_size=args.batch_size, num_workers = args.num_workers)
    print(f'The head has {sum(p.numel() for p in base_model[1].parameters())} parameters, the tail has {sum(p.numel() for p in base_model[2].parameters())} parameters.')

    # Place to save the trained model
    save_dir = f'saved_models/{args.model_name}-{args.dataset_name}'
    os.makedirs(save_dir, exist_ok = True)
    
    # Prepare for training
    optimizer = torch.optim.Adam(base_model.parameters(), lr = args.learning_rate)
    Loss = nn.CrossEntropyLoss()
    
    # training
    best_accuracy = 0.0
    for n in range(args.num_epochs):

        base_model.train()
        epoch_loss = 0.0
        for X, y in training_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = Loss(base_model(X), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(y) / len(training_set)

        # testing
        base_model.eval()
        accuracy = 0.0
        with torch.no_grad():
            for X, y in testing_loader:
                X, y = X.to(device), y.to(device)
                _, pred = base_model(X).max(axis = -1)
                accuracy += (pred == y).sum().item() / len(testing_set)

        print(f'Epoch {n}, loss {epoch_loss:.3f}, accuracy = {accuracy:.4f}.')

        # save the best result
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(base_model[1].state_dict(), f'{save_dir}/base_head_state_dict')
            torch.save(base_model[2].state_dict(), f'{save_dir}/base_tail_state_dict')

    print(f'Completed the training of the base model, {args.model_name}-{args.dataset_name}, accuracy = {best_accuracy:.4f}.')
