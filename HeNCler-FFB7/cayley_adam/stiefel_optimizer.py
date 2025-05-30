# from .optimizer import Optimizer, required
import random

import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required

from cayley_adam.gutils import Cayley_loop
from cayley_adam.gutils import qr_retraction
from cayley_adam.gutils import unit
from cayley_adam.utils import matrix_norm_one

episilon = 1e-8


class SGDG(Optimizer):
    r"""This optimizer updates variables with two different routines
        based on the boolean variable 'stiefel'. 

        If stiefel is True, the variables will be updated by SGD-G proposed 
        as decorrelated weight matrix.

        If stiefel is False, the variables will be updated by SGD.
        This routine was taken from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

        -- common parameters
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        stiefel (bool, optional): whether to use SGD-G (default: False)

        -- parameters in case stiefel is False 
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

        -- parameters in case stiefel is True
        omega (float, optional): orthogonality regularization factor (default: 0)
        grad_clip (float, optional): threshold for gradient norm clipping (default: None)
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,
                 stiefel=False, omega=0, grad_clip=None):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        stiefel=stiefel, omega=0, grad_clip=grad_clip)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires adj momentum and zero dampening")
        super(SGDG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs adj single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            stiefel = group['stiefel']

            for p in group['params']:
                if p.grad is None:
                    continue

                unity, _ = unit(p.data.view(p.size()[0], -1))
                if stiefel and unity.size()[0] <= unity.size()[1]:

                    weight_decay = group['weight_decay']
                    dampening = group['dampening']
                    nesterov = group['nesterov']

                    rand_num = random.randint(1, 101)
                    if rand_num == 1:
                        unity = qr_retraction(unity)

                    g = p.grad.data.view(p.size()[0], -1)

                    lr = group['lr']

                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = torch.zeros(g.t().size())
                        if p.is_cuda:
                            param_state['momentum_buffer'] = param_state['momentum_buffer'].cuda()

                    V = param_state['momentum_buffer']
                    V = momentum * V - g.t()
                    MX = torch.mm(V, unity)
                    XMX = torch.mm(unity, MX)
                    XXMX = torch.mm(unity.t(), XMX)
                    W_hat = MX - 0.5 * XXMX
                    W = W_hat - W_hat.t()
                    t = 0.5 * 2 / (matrix_norm_one(W) + episilon)
                    alpha = min(t, lr)

                    p_new = Cayley_loop(unity.t(), W, V, alpha)
                    V_new = torch.mm(W, unity.t())  # n-by-p
                    #                     check_identity(p_new.t())
                    p.data.copy_(p_new.view(p.size()))
                    V.copy_(V_new)

                else:
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = d_p.clone()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                    p.data.add_(-group['lr'], d_p)

        return loss


class AdamG(Optimizer):
    r"""This optimizer updates variables with two different routines
        based on the boolean variable 'grassmann'. 

        If grassmann is True, the variables will be updated by Adam-G proposed 
        in 'Riemannian approach to batch normalization'.

        If grassmann is False, the variables will be updated by SGD.
        This routine was taken from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py.


    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

        -- common parameters
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        grassmann (bool, optional): whether to use Adam-G (default: False)

        -- parameters in case grassmann is False 
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

        -- parameters in case grassmann is True
        beta2 (float, optional): the exponential decay rate for the second moment estimates (defulat: 0.99)
        epsilon (float, optional): adj small constant for numerical stability (default: 1e-8)
        omega (float, optional): orthogonality regularization factor (default: 0)
        grad_clip (float, optional): threshold for gradient norm clipping (default: None)
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,
                 grassmann=False, beta2=0.99, epsilon=1e-8, omega=0, grad_clip=None):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        grassmann=grassmann, beta2=beta2, epsilon=epsilon, omega=0, grad_clip=grad_clip)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires adj momentum and zero dampening")
        super(AdamG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs adj single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            stiefel = group['stiefel']

            for p in group['params']:
                if p.grad is None:
                    continue

                beta1 = group['momentum']
                beta2 = group['beta2']
                epsilon = group['epsilon']

                unity, _ = unit(p.data.view(p.size()[0], -1))
                if stiefel and unity.size()[0] <= unity.size()[1]:
                    rand_num = random.randint(1, 101)
                    if rand_num == 1:
                        unity = qr_retraction(unity)

                    g = p.grad.data.view(p.size()[0], -1)

                    param_state = self.state[p]
                    if 'm_buffer' not in param_state:
                        size = p.size()
                        param_state['m_buffer'] = torch.zeros([int(np.prod(size[1:])), size[0]])
                        param_state['v_buffer'] = torch.zeros([1])
                        if p.is_cuda:
                            param_state['m_buffer'] = param_state['m_buffer'].cuda()
                            param_state['v_buffer'] = param_state['v_buffer'].cuda()

                        param_state['beta1_power'] = beta1
                        param_state['beta2_power'] = beta2

                    m = param_state['m_buffer']
                    v = param_state['v_buffer']
                    beta1_power = param_state['beta1_power']
                    beta2_power = param_state['beta2_power']

                    mnew = beta1 * m + (1.0 - beta1) * g.t()  # p by n
                    vnew = beta2 * v + (1.0 - beta2) * (torch.norm(g) ** 2)

                    mnew_hat = mnew / (1 - beta1_power)
                    vnew_hat = vnew / (1 - beta2_power)

                    MX = torch.matmul(mnew_hat, unity)
                    XMX = torch.matmul(unity, MX)
                    XXMX = torch.matmul(unity.t(), XMX)
                    W_hat = MX - 0.5 * XXMX
                    W = (W_hat - W_hat.t()) / vnew_hat.add(epsilon).sqrt()

                    t = 0.5 * 2 / (matrix_norm_one(W) + episilon)
                    alpha = min(t, group['lr'])

                    p_new = Cayley_loop(unity.t(), W, mnew, -alpha)

                    p.data.copy_(p_new.view(p.size()))
                    mnew = torch.matmul(W, unity.t()) * vnew_hat.add(epsilon).sqrt() * (1 - beta1_power)
                    m.copy_(mnew)
                    v.copy_(vnew)

                    param_state['beta1_power'] *= beta1
                    param_state['beta2_power'] *= beta2

                else:
                    momentum = group['momentum']
                    weight_decay = group['weight_decay']
                    dampening = group['dampening']
                    nesterov = group['nesterov']
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = d_p.clone()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                    # p.data.add_(-group['lr'], d_p)
                    p.data.add_(d_p, alpha=-group['lr'])
        return loss
