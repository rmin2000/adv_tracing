from attacks import *

import torch



class DecisionBlackBoxAttack(object):
    def __init__(self, max_queries=np.inf, epsilon=0.5, p='inf', lb=0., ub=1., batch_size=1):
        """
        :param max_queries: max number of calls to model per data point
        :param epsilon: perturbation limit according to lp-ball
        :param p: norm for the lp-ball constraint
        :param lb: minimum value data point can take in any coordinate
        :param ub: maximum value data point can take in any coordinate
        """
        assert p in ['inf', '2'], "L-{} is not supported".format(p)

        self.p = p
        self.max_queries = max_queries
        self.total_queries = 0
        self.total_successes = 0
        self.total_failures = 0
        self.total_distance = 0
        self.sigma = 0
        self.EOT = 1
        self.lb = lb
        self.ub = ub
        self.epsilon = epsilon / ub
        self.batch_size = batch_size
        self.list_loss_queries = torch.zeros(1, self.batch_size)

    def result(self):
        """
        returns a summary of the attack results (to be tabulated)
        :return:
        """
        list_loss_queries = self.list_loss_queries[1:].view(-1)
        mask = list_loss_queries > 0
        list_loss_queries = list_loss_queries[mask]
        self.total_queries = int(self.total_queries)
        self.total_successes = int(self.total_successes)
        self.total_failures = int(self.total_failures)
        return {
            "total_queries": self.total_queries,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "average_num_queries": "NaN" if self.total_successes == 0 else self.total_queries / self.total_successes,
            "failure_rate": "NaN" if self.total_successes + self.total_failures == 0 else self.total_failures / (self.total_successes + self.total_failures),
            "median_num_loss_queries": "NaN" if self.total_successes == 0 else torch.median(list_loss_queries).item(), 
            "config": self._config()
        }

    def _config(self):
        """
        return the attack's parameter configurations as a dict
        :return:
        """
        raise NotImplementedError

    def distance(self, x_adv, x = None):
        if x is None:
            diff = x_adv.view(x_adv.shape[0], -1)
        else:
            diff = (x_adv - x).view(x.shape[0], -1)
        if self.p == '2':
            out = torch.sqrt(torch.sum(diff * diff, dim = 1))
        elif self.p == 'inf':
            out, _ = torch.max(torch.abs(diff), dim = 1)
        return out
    
    def is_adversarial(self, x, y):
        '''
        check whether the adversarial constrain holds for x
        '''
        if self.targeted:
            return self.predict_label(x) == y
        else:
            return self.predict_label(x) != y

    def predict_label(self, xs):
        with torch.no_grad():
            if type(xs) is torch.Tensor:
                out = self.model(xs).argmax(dim=-1).squeeze()
            else:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                out = self.model(torch.FloatTensor(xs).to(device)).argmax(dim=-1).squeeze()
            return out

    def _perturb(self, xs_t, ys):
        raise NotImplementedError

    def run(self, Xs, ys, model, targeted, dset): 
        self.model = model
        self.targeted = targeted

        X_attacked = []

        for x, y in zip(Xs, ys):
            adv, _ = self._perturb(x[None, ...], y[None])
            X_attacked.append(adv.squeeze())
        X_attacked = torch.stack(X_attacked).float()

        success = (self.distance(X_attacked,Xs) < self.epsilon)
        
        return X_attacked * success[:, None, None, None] + Xs * (~success[:, None, None, None])
