import numpy as np

import torch

from torch import Tensor as t

from attacks import *

class ScoreBlackBoxAttack(object):
    def __init__(self, max_loss_queries=np.inf,
                 max_extra_queries=np.inf,
                 epsilon=0.5, p='inf', lb=0., ub=1.,batch_size = 50, name = '', device = 'cuda'):
        """
        :param max_loss_queries: max number of calls to model per data point
        :param max_extra_queries: max number of calls to early stopping extraerion per data point
        :param epsilon: perturbation limit according to lp-ball
        :param p: norm for the lp-ball constraint
        :param lb: minimum value data point can take in any coordinate
        :param ub: maximum value data point can take in any coordinate
        """
        assert p in ['inf', '2'], "L-{} is not supported".format(p)

        self.epsilon = epsilon
        self.p = p
        self.batch_size = batch_size
        self.max_loss_queries = max_loss_queries
        self.max_extra_queries = max_extra_queries
        self.list_loss_queries = torch.zeros(1, self.batch_size, device = device)
        self.total_loss_queries = 0
        self.total_extra_queries = 0
        self.total_successes = 0
        self.total_failures = 0
        self.lb = lb
        self.ub = ub
        self.name = name
        # the _proj method takes pts and project them into the constraint set:
        # which are
        #  1. epsilon lp-ball around xs
        #  2. valid data pt range [lb, ub]
        # it is meant to be used within `self.run` and `self._perturb`
        self._proj = None
        # a handy flag for _perturb method to denote whether the provided xs is a
        # new batch (i.e. the first iteration within `self.run`)
        self.is_new_batch = False

    def result(self):
        """
        returns a summary of the attack results (to be tabulated)
        :return:
        """
        list_loss_queries = self.list_loss_queries[1:].view(-1)
        mask = list_loss_queries > 0
        list_loss_queries = list_loss_queries[mask]
        self.total_loss_queries = int(self.total_loss_queries)
        self.total_extra_queries = int(self.total_extra_queries)
        self.total_successes = int(self.total_successes)
        self.total_failures = int(self.total_failures)
        return {
            "total_loss_queries": self.total_loss_queries,
            "total_extra_queries": self.total_extra_queries,
            "average_num_loss_queries": "NaN" if self.total_successes == 0 else self.total_loss_queries / self.total_successes,
            "average_num_extra_queries": "NaN" if self.total_successes == 0 else self.total_extra_queries / self.total_successes,
            "median_num_loss_queries": "NaN" if self.total_successes == 0 else torch.median(list_loss_queries).item(), 
            "total_queries": self.total_extra_queries + self.total_loss_queries,
            "average_num_queries": "NaN" if self.total_successes == 0 else (self.total_extra_queries + self.total_loss_queries) / self.total_successes,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "failure_rate": "NaN" if self.total_successes + self.total_failures == 0 else self.total_failures / (self.total_successes + self.total_failures),
            "config": self._config()
        }

    def _config(self):
        """
        return the attack's parameter configurations as a dict
        :return:
        """
        raise NotImplementedError

    def _perturb(self, xs_t, loss_fct):
        """
        :param xs_t: batch_size x dim x .. (torch tensor)
        :param loss_fct: function to query (the attacker would like to maximize) (batch_size data pts -> R^{batch_size}
        :return: suggested xs as a (torch tensor)and the used number of queries per data point
            i.e. a tuple of (batch_size x dim x .. tensor, batch_size array of number queries used)
        """
        raise NotImplementedError

    def proj_replace(self, xs_t, sugg_xs_t, dones_mask_t):
        sugg_xs_t = self._proj(sugg_xs_t)
        # replace xs only if not done
        xs_t = sugg_xs_t * (1. - dones_mask_t) + xs_t * dones_mask_t
        return xs_t

    def run(self, xs, loss_fct, early_stop_extra_fct):
        """
        attack with `xs` as data points using the oracle `l` and
        the early stopping extraerion `early_stop_extra_fct`
        :param xs: data points to be perturbed adversarially (numpy array)
        :param loss_fct: loss function (m data pts -> R^m)
        :param early_stop_extra_fct: early stop function (m data pts -> {0,1}^m)
                ith entry is 1 if the ith data point is misclassified
        :return: a dict of logs whose length is the number of iterations
        """
        # convert to tensor
        xs_t = torch.clone(xs)
        
        batch_size = xs.shape[0]
        num_axes = len(xs.shape[1:])
        num_loss_queries = torch.zeros(batch_size, device = xs.device)
        num_extra_queries = torch.zeros(batch_size, device = xs.device)

        dones_mask = early_stop_extra_fct(xs_t)
        correct_classified_mask = ~dones_mask

        # init losses for performance tracking
        losses = torch.zeros(batch_size, device = xs.device)

        # make a projector into xs lp-ball and within valid pixel range
        if self.p == '2':
            _proj = l2_proj_maker(xs_t, self.epsilon)
            self._proj = lambda _: torch.clamp(_proj(_), self.lb, self.ub)
        elif self.p == 'inf':
            _proj = linf_proj_maker(xs_t, self.epsilon)
            self._proj = lambda _: torch.clamp(_proj(_), self.lb, self.ub)
        else:
            raise Exception('Undefined l-p!')

        # iterate till model evasion or budget exhaustion
        self.is_new_batch = True
        its = 0
        while True:
            # if np.any(num_loss_queries + num_extra_queries >= self.max_loss_queries):
            if torch.any(num_loss_queries >= self.max_loss_queries):
                print("#loss queries exceeded budget, exiting")
                break
            if torch.any(num_extra_queries >= self.max_extra_queries):
                print("#extra_queries exceeded budget, exiting")
                break
            if torch.all(dones_mask):
                print("all data pts are misclassified, exiting")
                break
            # propose new perturbations
            sugg_xs_t, num_loss_queries_per_step = self._perturb(xs_t, loss_fct)
            # project around xs and within pixel range and
            # replace xs only if not done
            ##updated x here
            xs_t = self.proj_replace(xs_t, sugg_xs_t, (dones_mask.reshape(-1, *[1] * num_axes).float()))

            # update number of queries (note this is done before updating dones_mask)
            num_loss_queries += num_loss_queries_per_step * (~dones_mask)
            num_extra_queries += (~dones_mask)
            losses = loss_fct(xs_t) * (~dones_mask) + losses * dones_mask

            # update dones mask
            dones_mask = dones_mask | early_stop_extra_fct(xs_t)
            success_mask = dones_mask * correct_classified_mask
            its += 1

            self.is_new_batch = False            


        success_mask = dones_mask * correct_classified_mask
        self.total_loss_queries += (num_loss_queries * success_mask).sum()
        self.total_extra_queries += (num_extra_queries * success_mask).sum()
        self.list_loss_queries = torch.cat([self.list_loss_queries, torch.zeros(1, batch_size, device = xs.device)], dim=0)
        self.list_loss_queries[-1] = num_loss_queries * success_mask
        self.total_successes += success_mask.sum()
        self.total_failures += ((~dones_mask) * correct_classified_mask).sum()

        # set self._proj to None to ensure it is intended use
        self._proj = None

        return xs_t
