"""ATTCG optimizer.
    # Arguments
        learning_rate: float >= 0. Learning rate. default 0.001
        beta: float, 0 < beta < 1. Generally close to 1. default 0.999
"""

import math
import torch
from torch.optim.optimizer import Optimizer


class ATTCG(Optimizer):

    def __init__(self, params, lr=1e-3, beta=0.999):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, beta=beta)
        super(ATTCG, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('ATTCG does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['eta'] = group['lr']
                    state['mt'] = 1e-8 * torch.ones_like(p, memory_format=torch.preserve_format)
                    state['vt'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['og'] = 1e-8 * torch.ones_like(grad, memory_format=torch.preserve_format)
                    state['amsgrad'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # Update the steps for each param group update
                state['step'] += 1
                amsgrad = state['amsgrad']

                pastck = (torch.sqrt(torch.sum(torch.pow(state['og'], 2))) * torch.sqrt(torch.sum(torch.pow(state['og'], 2))))
                if pastck < 1e-16:
                    pastck = 1e-16

                yk = grad.sub(state['og'])
                betaprp = grad * yk / pastck
                theta = grad * state['mt'] / pastck

                state['mt'] = -grad + betaprp * state['mt'] - theta * yk
                state['vt'] = group['beta'] * state['vt'].add_((1-group['beta']) * torch.square(state['mt']), alpha=1)
                torch.maximum(amsgrad, state['vt'], out=amsgrad)

                lr_t = state['eta'] * (math.sqrt(1. - math.pow(group['beta'], state['step'])))
                posts = 1 / (torch.sqrt(amsgrad) + 1e-8)
                step = posts * lr_t
                p.add_(state['mt'] * step, alpha=1)
                state['og'].copy_(grad)

            return

        return loss

