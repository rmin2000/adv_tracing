import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
import argparse
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import HopSkipJump

from models import VGG16Head, VGG16Tail, ResNet18Head, ResNet18Tail
import config
from watermark import Watermark



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help = 'Benchmark model structure.', choices = ['VGG16', 'ResNet18'])
    parser.add_argument('--dataset_name', help = 'Benchmark dataset used.', choices = ['CIFAR10', 'GTSRB'])
    parser.add_argument('-M', '--num_models', help = 'The number of models used.', type = int, default = 100)
    parser.add_argument('-n', '--num_samples', help = 'The number of adversarial samples per model.', type = int, default = 1)
    parser.add_argument('-c', '--cont', help = 'Continue from the stopped point last time.', action = 'store_true')
    parser.add_argument('-b', '--batch_size', help = 'The batch size used for attacks.', type = int, default = 10)
    parser.add_argument('-v', '--verbose', help = 'Verbose when attacking.', action = 'store_true')
    args = parser.parse_args()
    
    dataset = eval(f'config.{args.dataset_name}()')
    training_set, testing_set = dataset.training_set, dataset.testing_set
    num_classes = dataset.num_classes
    means, stds = dataset.means, dataset.stds
    C, H, W = dataset.C, dataset.H, dataset.W
    Head, Tail = eval(f'{args.model_name}Head'), eval(f'{args.model_name}Tail')
    testing_loader = torch.utils.data.DataLoader(testing_set, batch_size = args.batch_size, shuffle = True, num_workers = 2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_dir = f'saved_models/{args.model_name}-{args.dataset_name}'

    save_dir = f'saved_adv_examples/{args.model_name}-{args.dataset_name}'


    # load the tail of the model
    normalizer = transforms.Normalize(means, stds)
    tail = Tail(num_classes)
    tail.load_state_dict(torch.load(f'{model_dir}/base_tail_state_dict'))
    tail.to(device)

    # load the classifiers
    classifiers = []
    models = []
    for i in range(args.num_models):
        head = Head()
        head.to(device)
        head.load_state_dict(torch.load(f'{model_dir}/head_{i}/state_dict'))
        watermark = Watermark.load(f'{model_dir}/head_{i}/watermark.npy')
        models.append(nn.Sequential(normalizer, watermark, head, tail, nn.Softmax(dim = -1)).eval())
        models[-1].to(device)
        classifier = PyTorchClassifier(
            model = nn.Sequential(normalizer, watermark, head, tail, nn.Softmax(dim = -1)).eval(),
            loss = None,
            optimizer = None,
            clip_values = (0, 1),
            input_shape=(C, H, W),
            nb_classes=num_classes,
            device_type = 'gpu' if torch.cuda.is_available() else 'cpu'
        )
        classifiers.append(classifier)
    classifiers = np.array(classifiers)

    # attacking
    for i, (model, c) in enumerate(zip(models, classifiers)):
        a = HopSkipJump(c, verbose = args.verbose)
        if os.path.isfile(f'{save_dir}/head_{i}/HopSkipJump_proj.npz') and args.cont:
            continue

        original_images, attacked_images, labels = [], [], []
        count_success = 0

        for X, y in testing_loader:
            X, y = X.numpy(), y.numpy()
            pred = c.predict(X)
            correct_mask = pred.argmax(axis = 1) == y

            X_attacked = a.generate(X)
            attacked_preds = np.vectorize(lambda z: z.predict(X_attacked), signature = '()->(m,n)')(classifiers) # (num_model, batch_size, num_class)
            success_mask = attacked_preds.argmax(axis = -1) != y 
            success_mask = np.logical_and(success_mask[i], success_mask.sum(axis=0) >= 2)
            mask = np.logical_and(correct_mask, success_mask)
            
            original_images.append(X[mask])
            attacked_images.append(X_attacked[mask])
            labels.append(y[mask])
            
            count_success += mask.sum()
            if count_success >= args.num_samples:
                print(f'Head {i}, attack HopSkipJump, {count_success} out of {args.num_samples} generated, done!')
                break
            else:
                print(f'Head {i}, attack HopSkipJump, {count_success} out of {args.num_samples} generated...')
            
        original_images = np.concatenate(original_images)
        attacked_images = np.concatenate(attacked_images)
        labels = np.concatenate(labels)
        os.makedirs(f'{save_dir}/head_{i}', exist_ok = True)
        np.savez(f'{save_dir}/head_{i}/HopSkipJump.npz', X = original_images, X_attacked = attacked_images, y = labels)
