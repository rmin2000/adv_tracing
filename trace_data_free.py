import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import argparse
import logging
from art.estimators.classification import PyTorchClassifier

from models import VGG16Head, VGG16Tail, ResNet18Head, ResNet18Tail
import config
from watermark import Watermark


def get_classifier(watermark, model, means, stds, num_class):
    return PyTorchClassifier(
        model = nn.Sequential(transforms.Normalize(means, stds), watermark, model, nn.Softmax(dim = -1)).eval(),
        loss = None, # dummy
        optimizer = None, # dummy
        input_shape=(C, H, W),
        clip_values = (0, 1),
        nb_classes=num_class,
        device_type = 'gpu' if torch.cuda.is_available() else 'cpu'
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='ResNet18', help = 'Benchmark model structure.', choices = ['VGG16', 'ResNet18'])
    parser.add_argument('--dataset_name', default='CIFAR10', help = 'Benchmark dataset used.', choices = ['CIFAR10', 'GTSRB', 'TINY'])
    parser.add_argument('--attacks', default='Bandit', help = 'Attacks to be explored.', nargs = '+')
    parser.add_argument('--alpha', help = 'Hyper-parameter alpha.', type = float)
    parser.add_argument('-M', '--num_models', help = 'The number of models used for identification.', type = int, default = 50)
    parser.add_argument('-n', '--num_samples', help = 'The number of adversarial samples per model.', type = int, default = 1)
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
   

    dataset = eval(f'config.{args.dataset_name}()')
    training_set, testing_set = dataset.training_set, dataset.testing_set
    num_classes = dataset.num_classes
    means, stds = dataset.means, dataset.stds
    C, H, W = dataset.C, dataset.H, dataset.W
    Head, Tail = eval(f'{args.model_name}Head'), eval(f'{args.model_name}Tail')
    normalizer = transforms.Normalize(means, stds)

   
    model_dir = f'./saved_models/{args.model_name}-{args.dataset_name}'
    adv_dir = f'./saved_adv_examples/{args.model_name}-{args.dataset_name}'

    # load the tail of the model
    normalizer = transforms.Normalize(means, stds)
    tail = Tail(num_classes)
    tail.load_state_dict(torch.load(f'{model_dir}/base_tail_state_dict'))
    tail.to(device)
    tail.eval() 

    # load the classifiers
    heads, watermarks, models = [], [], []
    for i in range(args.num_models):
        heads.append(Head())
        heads[-1].to(device)
        heads[-1].load_state_dict(torch.load(f'{model_dir}/head_{i}/state_dict'))
        heads[-1].eval()
        watermarks.append(Watermark.load(f'{model_dir}/head_{i}/watermark.npy'))
        models.append(nn.Sequential(heads[-1], tail))
    
    for a in args.attacks:
        correct = 0
        Loss = nn.CrossEntropyLoss()
        for i in range(args.num_models):
            adv_npz = np.load(f'{adv_dir}/head_{i}/{a}.npz')
            X, X_attacked, y = adv_npz['X'][:args.num_samples], adv_npz['X_attacked'][:args.num_samples], adv_npz['y'][:args.num_samples]
            
            classifier_matrix = np.array([[get_classifier(wm, m, means, stds, num_classes) for wm in watermarks] for m in models])
            predictions = np.vectorize(lambda c: c.predict(X_attacked), signature='()->(m,n)')(classifier_matrix)
            
            X, X_attacked, y = torch.tensor(X).to(device), torch.tensor(X_attacked).to(device), torch.tensor(y).to(device)
            CE_loss = torch.stack([Loss(tail(head(wm(normalizer(X_attacked)))).softmax(-1), y) for wm, head in zip(watermarks, heads)], axis = 0).cpu()
            
            
            out = torch.stack([tail(head(wm(normalizer(X_attacked)))).argmax(axis = -1) for wm, head in zip(watermarks, heads)], axis = 0)
            wrong_pred = (out == y[None,:]).sum(-1) > 0
            
            predictions_maximum_class = predictions.max(axis = -1)
            
            maximum_class_score = predictions_maximum_class[np.arange(args.num_models), np.arange(args.num_models), ...] / predictions_maximum_class.sum(1)
            maximum_class_score = torch.from_numpy(maximum_class_score).sum(-1)
            
            score = maximum_class_score + args.alpha * CE_loss 
            score[wrong_pred]=np.inf
            result = score.topk(1, axis = 0, largest=False)[1]

            correct += torch.sum(result == i).item()
        print((f'Attack {a}, tracing accuracy {correct / args.num_models}.'))
