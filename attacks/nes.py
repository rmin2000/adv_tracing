import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
import argparse
from art.estimators.classification import PyTorchClassifier

from models import VGG16Head, VGG16Tail, ResNet18Head, ResNet18Tail
import config
from watermark import Watermark
from attacks.score import ScoreBlackBoxAttack
from attacks import *

Loss = nn.CrossEntropyLoss(reduction = 'none')

class NESAttack(ScoreBlackBoxAttack):
    """
    NES Attack
    """

    def __init__(self, max_loss_queries, epsilon, p, fd_eta, lr, q, lb, ub, batch_size, name):
        """
        :param max_loss_queries: maximum number of calls allowed to loss oracle per data pt
        :param epsilon: radius of lp-ball of perturbation
        :param p: specifies lp-norm  of perturbation
        :param fd_eta: forward difference step
        :param lr: learning rate of NES step
        :param q: number of noise samples per NES step
        :param lb: data lower bound
        :param ub: data upper bound
        """
        super().__init__(max_extra_queries=np.inf,
                         max_loss_queries=max_loss_queries,
                         epsilon=epsilon,
                         p=p,
                         lb=lb,
                         ub=ub,
                         batch_size= batch_size,
                         name = "NES")
        self.q = q
        self.fd_eta = fd_eta
        self.lr = lr

    def _perturb(self, xs_t, loss_fct):
        _shape = list(xs_t.shape)
        dim = np.prod(_shape[1:])
        num_axes = len(_shape[1:])
        gs_t = torch.zeros_like(xs_t)
        for _ in range(self.q):
            # exp_noise = torch.randn_like(xs_t) / (dim ** 0.5)
            exp_noise = torch.randn_like(xs_t)
            fxs_t = xs_t + self.fd_eta * exp_noise
            bxs_t = xs_t - self.fd_eta * exp_noise
            est_deriv = (loss_fct(fxs_t) - loss_fct(bxs_t)) / (4. * self.fd_eta)
            gs_t += est_deriv.reshape(-1, *[1] * num_axes) * exp_noise
        # perform the step
        new_xs = lp_step(xs_t, gs_t, self.lr, self.p)
        return new_xs, 2 * self.q * torch.ones(_shape[0], device = xs_t.device)

    def _config(self):
        return {
            "name": self.name,
            "p": self.p,
            "epsilon": self.epsilon,
            "lb": self.lb,
            "ub": self.ub,
            "max_extra_queries": "inf" if np.isinf(self.max_extra_queries) else self.max_extra_queries,
            "max_loss_queries": "inf" if np.isinf(self.max_loss_queries) else self.max_loss_queries,
            "lr": self.lr,
            "q": self.q,
            "fd_eta": self.fd_eta,
            "attack_name": self.__class__.__name__
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help = 'Benchmark model structure.', choices = ['VGG16', 'ResNet18'])
    parser.add_argument('--dataset_name', help = 'Benchmark dataset used.', choices = ['CIFAR10', 'GTSRB'])
    parser.add_argument('-M', '--num_models', help = 'The number of models used.', type = int, default = 100)
    parser.add_argument('-n', '--num_samples', help = 'The number of adversarial samples per model.', type = int, default = 1)
    parser.add_argument('-c', '--cont', help = 'Continue from the stopped point last time.', action = 'store_true')
    parser.add_argument('-b', '--batch_size', help = 'The batch size used for attacks.', type = int, default = 10)
    args = parser.parse_args()
    
    # renaming
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

        models.append(nn.Sequential(normalizer, watermark, head, tail).eval())
        models[-1].to(device)
        
        classifier = PyTorchClassifier(
            model = models[-1],
            loss = None,
            optimizer = None,
            clip_values = (0, 1),
            input_shape=(C, H, W),
            nb_classes=num_classes,
            device_type = 'gpu' if torch.cuda.is_available() else 'cpu'
        )
        classifiers.append(classifier)
    classifiers = np.array(classifiers)

    for i, (model, c) in enumerate(zip(models, classifiers)):
        if os.path.isfile(f'{save_dir}/head_{i}/NES.npz') and args.cont:
            continue
        original_images, attacked_images, labels = [], [], []
        count_success = 0
        for X, y in testing_loader:
            with torch.no_grad():
                pred = c.predict(X.numpy())
                correct_mask = pred.argmax(axis = -1) == y.numpy()

                X_device, y_device = X.to(device), y.to(device)
                def loss_fct(xs, es = False):
                    logits = model(xs)
                    loss = Loss(logits.to(device), y_device)
                    if es:
                        return torch.argmax(logits, axis= -1) != y_device, loss
                    else: 
                        return loss

                def early_stop_crit_fct(xs):
                    logits = model(xs)
                    return logits.argmax(axis = -1) != y_device

                a = NESAttack(max_loss_queries = 10000, epsilon = 1.0, p = '2', fd_eta = 0.01, lr = 0.01, q = 15, lb = 0.0, ub = 1.0, batch_size = args.batch_size, name = 'NESAttack')

                X_attacked = a.run(X_device, loss_fct, early_stop_crit_fct).cpu().numpy()

                attacked_preds = np.vectorize(lambda z: z.predict(X_attacked), signature = '()->(m,n)')(classifiers)
                
                success_mask = attacked_preds.argmax(axis = -1) != y.numpy()
                success_mask = np.logical_and(success_mask[i], success_mask.sum(axis=0) >= 2)

                mask = np.logical_and(correct_mask, success_mask)
                
                original_images.append(X[mask])
                attacked_images.append(X_attacked[mask])
                labels.append(y[mask])
                
                count_success += mask.sum()
                if count_success >= args.num_samples:
                    print(f'Model {i}, attack NES, {count_success} out of {args.num_samples} generated, done!')
                    break
                else:
                    print(f'Model {i}, attack NES, {count_success} out of {args.num_samples} generated...')
        
        original_images = np.concatenate(original_images)
        attacked_images = np.concatenate(attacked_images)
        labels = np.concatenate(labels)
        os.makedirs(f'{save_dir}/head_{i}', exist_ok = True)
        np.savez(f'{save_dir}/head_{i}/NES.npz', X = original_images, X_attacked = attacked_images, y = labels)
