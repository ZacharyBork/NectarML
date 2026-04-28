from __future__ import annotations

from typing import Any

import _nectarml
from nectarml.core            import Tensor, creation as T
from nectarml.optim.optimizer import Optimizer

class Adam(Optimizer):
    def __init__(
        self: Adam,
        parameters: (
            list[Tensor] 
          | list[tuple[str, Tensor]]
          | list[dict[str, Any]]
        ),
        lr:                    float = 0.003,
        betas:   tuple[float, float] = (0.9, 0.999),
        eps:                   float = 1e-8,
        weight_decay:          float = 0.0,
        decoupled_weight_decay: bool = False,
        amsgrad:                bool = False,
        maximize:               bool = False,
        foreach:                bool = None,  # NOT YET IMPLEMENTED
        capturable:             bool = False, # NOT YET IMPLEMENTED
        fused:                  bool = True
    ) -> None:
        '''
        Reference:
            - Diederik P. Kingma, and Jimmy Lei Ba, "Adam: A Method for 
                Stochastic Optimization.", https://arxiv.org/pdf/1412.6980
        '''
        super().__init__(parameters, defaults={
            'lr':                     lr,
            'betas':                  betas,
            'eps':                    eps,
            'weight_decay':           weight_decay,
            'decoupled_weight_decay': decoupled_weight_decay,
            'amsgrad':                amsgrad
        })
        self.lr                     = lr
        self.beta1, self.beta2      = betas
        self.eps                    = eps
        self.weight_decay           = weight_decay
        self.decoupled_weight_decay = decoupled_weight_decay
        self.amsgrad                = amsgrad
        self.maximize               = maximize
        self.foreach                = foreach
        self.capturable             = capturable
        self.fused                  = fused
              
    def _build_state(self: Adam, param_index: int, param: Tensor) -> None:
        if param_index not in self.state: self.state[param_index] = {}
        if 'exp_avg' not in self.state[param_index]:
            self.state[param_index]['exp_avg'] = T.zeros_like(param)
        if 'exp_avg_sq' not in self.state[param_index]:
            self.state[param_index]['exp_avg_sq'] = T.zeros_like(param)
        if 'step' not in self.state[param_index]: 
            self.state[param_index]['step'] = 0
        if self.amsgrad:
            if 'max_exp_avg_sq' not in self.state[param_index]:
                self.state[param_index]['max_exp_avg_sq'] = T.zeros_like(param)
      
    def _update_step(self: Adam, param_index: int) -> int:
        self.state[param_index]['step'] += 1
        return self.state[param_index]['step']
      
    def _apply_decay(
        self:  Adam, 
        param: Tensor, 
        grad:  Tensor,
        lr:    float
    ) -> tuple[Tensor, Tensor]:
        if self.maximize: grad = -grad
        if self.weight_decay: 
            if self.decoupled_weight_decay:
                  param -= lr   * self.weight_decay * param.detach()
            else: grad   = grad + self.weight_decay * param.detach()
        return param, grad
      
    def _update_averages(
        self: Adam, 
        idx:  int, 
        grad: Tensor
    ) -> tuple[Tensor, Tensor]:
        exp_avg = self.state[idx]['exp_avg']
        exp_avg = self.beta1 * exp_avg + (1 - self.beta1) * grad
        self.state[idx]['exp_avg'] = exp_avg.detach()
        
        exp_avg_sq = self.state[idx]['exp_avg_sq']
        exp_avg_sq = self.beta2 * exp_avg_sq + (1-self.beta2) * grad**2
        self.state[idx]['exp_avg_sq'] = exp_avg_sq.detach()
        
        return exp_avg, exp_avg_sq
      
    def _run_update(
        self:  Adam,
        param: Tensor,
        idx:   int,
        lr:    float,
        bc1:   float,
        bc2:   float
    ) -> None:
        grad = param.grad.detach().clone()
        
        param,   grad       = self._apply_decay(param, grad, lr)
        exp_avg, exp_avg_sq = self._update_averages(idx, grad)
        
        exp_avg_corr    = exp_avg    / bc1
        exp_avg_sq_corr = exp_avg_sq / bc2
        
        if self.amsgrad:
            max_exp_avg_sq = self.state[idx]['max_exp_avg_sq'].detach()
            max_exp_avg_sq = max_exp_avg_sq.maximum(exp_avg_sq_corr)
            self.state[idx]['max_exp_avg_sq'] = max_exp_avg_sq.detach()
            denom = max_exp_avg_sq.sqrt() + self.eps
        else: denom = exp_avg_sq_corr.sqrt() + self.eps
        
        param -= (lr * exp_avg_corr / denom).detach()

    def _update(self: Adam) -> None:
        for group in self.param_groups:
            _lr = group.get('lr', self.lr)
            for param in group['params']:
                if param.grad is None: continue
                idx = self._get_parameter_state_index(param)
                self._build_state(idx, param)
                
                step = self._update_step(idx)
                
                bias_correction1 = 1.0 - self.beta1**step
                bias_correction2 = 1.0 - self.beta2**step
                
                if self.fused and param.device == 'cuda':
                    max_exp_avg_sqrt = (
                        self.state[idx]['max_exp_avg_sq']._data_ptr
                        if self.amsgrad else 0)
                    _nectarml.optim.adam_update(
                        param._data_ptr, param.grad._data_ptr,
                        self.state[idx]['exp_avg']._data_ptr,
                        self.state[idx]['exp_avg_sq']._data_ptr,
                        max_exp_avg_sqrt, _lr, self.beta1, self.beta2, 
                        self.eps, bias_correction1, bias_correction2,
                        self.weight_decay, self.decoupled_weight_decay,
                        self.amsgrad, self.maximize, param.size)
                else:
                    self._run_update(
                        param, idx, _lr, bias_correction1, bias_correction2)

class AdamW(Adam):
    def __init__(
        self: AdamW,
        parameters: (
            list[Tensor] 
          | list[tuple[str, Tensor]]
          | list[dict[str, Any]]
        ),
        lr:                  float = 0.003,
        betas: tuple[float, float] = (0.9, 0.999),
        eps:                 float = 1e-8,
        weight_decay:        float = 0.0,
        amsgrad:              bool = False,
        maximize:             bool = False,
        foreach:              bool = None,  # NOT YET IMPLEMENTED
        capturable:           bool = False, # NOT YET IMPLEMENTED
        fused:                bool = True
    ) -> None:
        '''
        Reference:
            - Ilya Loshchilov, and Frank Hutter, "Decoupled Weight Decay 
                Regularization.", https://arxiv.org/pdf/1711.05101
        '''
        super().__init__(
            parameters, lr, betas, eps, weight_decay, True, 
            amsgrad, maximize, foreach, capturable, fused)

class NAdam(Adam):
    def __init__(
        self: NAdam,
        parameters: (
            list[Tensor] 
          | list[tuple[str, Tensor]]
          | list[dict[str, Any]]
        ),
        lr:                    float = 0.002,
        betas:   tuple[float, float] = (0.9, 0.999),
        eps:                   float = 1e-8,
        weight_decay:          float = 0.0,
        momentum_decay:        float = 0.004,
        decoupled_weight_decay: bool = False,
        maximize:               bool = False,
        foreach:                bool = None,  # NOT YET IMPLEMENTED
        capturable:             bool = False, # NOT YET IMPLEMENTED
        fused:                  bool = True
    ) -> None:
        '''
        Reference:
            - Dozat, Timothy, "Incorporating Nesterov Momentum into Adam.",
                https://cs229.stanford.edu/proj2015/054_report.pdf
        '''
        Optimizer.__init__(parameters, defaults={
            'lr':                     lr,
            'betas':                  betas,
            'eps':                    eps,
            'weight_decay':           weight_decay,
            'momentum_decay':         momentum_decay,
            'decoupled_weight_decay': decoupled_weight_decay
        })
        self.lr                     = lr
        self.beta1, self.beta2      = betas
        self.eps                    = eps
        self.weight_decay           = weight_decay
        self.momentum_decay         = momentum_decay
        self.decoupled_weight_decay = decoupled_weight_decay
        self.maximize               = maximize
        self.foreach                = foreach
        self.capturable             = capturable
        self.fused                  = fused
        
        super().__init__(
            parameters, lr, betas, eps, weight_decay, decoupled_weight_decay,
            False, maximize, foreach, capturable, fused)
        
        
    def _build_state(self: NAdam, param_index: int, param: Tensor) -> None:
        super()._build_state(param_index, param)
        if 'mu_product' not in self.state[param_index]:
            self.state[param_index]['mu_product'] = 1.0
    
    def _compute_momentum_decay(
        self:        NAdam, 
        step:        int, 
        param_index: int
    ) -> None:
        '''
        Based heavily on PyTorch's NAdam momentum decay:
        - https://github.com/pytorch/pytorch/blob/main/torch/optim/nadam.py
        '''
        self.mu_t   = self.beta1 \
                    * (1.0 - 0.5 * (0.96**(step * self.momentum_decay)))
        self.mu_t1  = self.beta1 \
                    * (1.0 - 0.5 * (0.96**((step+1) * self.momentum_decay)))
        
        mu_product  = self.state[param_index].get('mu_product', 1.0)
        mu_product *= self.mu_t
        self.state[param_index]['mu_product'] = mu_product

    def _run_update(
        self:  NAdam,
        param: Tensor,
        idx:   int,
        lr:    float,
        bc2:   float
    ) -> None:
        grad = param.grad.detach().clone()
        
        param,   grad       = self._apply_decay(param, grad, lr)
        exp_avg, exp_avg_sq = self._update_averages(idx, grad)
        
        denom         = (exp_avg_sq / bc2).sqrt() + self.eps
        mu_product    = self.state[idx]['mu_product']
        mu_prod_next  = mu_product * self.mu_t1        
        
        grad_scale    = -lr * (1 - self.mu_t) / (1 - mu_product)
        exp_avg_scale = -lr * self.mu_t1      / (1 - mu_prod_next)
                        
        update = (grad * grad_scale + exp_avg * exp_avg_scale) / denom
        param += update.detach()

    def _update(self: NAdam) -> None:
        for group in self.param_groups:
            _lr = group.get('lr', self.lr)
            for param in group['params']:
                if param.grad is None: continue
                idx = self._get_parameter_state_index(param)
                self._build_state(idx, param)
                
                step = self._update_step(idx)
                
                bias_correction = 1.0 - self.beta2**step
                self._compute_momentum_decay(step, idx)
                
                if self.fused and param.device == 'cuda':
                    _nectarml.optim.nadam_update(
                        param._data_ptr, param.grad._data_ptr,
                        self.state[idx]['exp_avg']._data_ptr,
                        self.state[idx]['exp_avg_sq']._data_ptr,
                        _lr, self.beta1, self.beta2, 
                        self.eps, self.mu_t, self.mu_t1, 
                        self.state[idx]['mu_product'], bias_correction, 
                        self.weight_decay, self.decoupled_weight_decay, 
                        self.maximize, param.size)
                else: self._run_update(param, idx, _lr, bias_correction)

class RAdam(Optimizer):
    def __init__(
        self: RAdam,
        parameters: (
            list[Tensor] 
          | list[tuple[str, Tensor]]
          | list[dict[str, Any]]
        ),
        lr:                    float = 0.003,
        betas:   tuple[float, float] = (0.9, 0.999),
        eps:                   float = 1e-8,
        weight_decay:          float = 0.0,
        decoupled_weight_decay: bool = False,
        maximize:               bool = False,
        foreach:                bool = None, # NOT YET IMPLEMENTED
        capturable:             bool = False # NOT YET IMPLEMENTED
    ) -> None:
        super().__init__(parameters, defaults={
            'lr':                     lr,
            'betas':                  betas,
            'eps':                    eps,
            'weight_decay':           weight_decay,
            'decoupled_weight_decay': decoupled_weight_decay
        })
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.decoupled_weight_decay = decoupled_weight_decay
        self.maximize = maximize
        self.foreach = foreach
        self.capturable = capturable
        
    def _update(self: RAdam) -> None:
        pass

class Adamax(Optimizer):
    def __init__(
        self: Adamax,
        parameters: (
            list[Tensor] 
          | list[tuple[str, Tensor]]
          | list[dict[str, Any]]
        ),
        lr:                  float = 0.003,
        betas: tuple[float, float] = (0.9, 0.999),
        eps:                 float = 1e-8,
        weight_decay:        float = 0.0,
        maximize:             bool = False,
        foreach:              bool = None, # NOT YET IMPLEMENTED
        capturable:           bool = False # NOT YET IMPLEMENTED
    ) -> None:
        super().__init__(
            parameters, 
            defaults={
                'lr': 0.003,
                'betas': (0.9, 0.999),
                'eps': 1e-8,
                'weight_decay': 0.0
            }
        )
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.maximize = maximize
        self.foreach = foreach
        self.capturable = capturable
        
    def _update(self: Adamax) -> None:
        pass




