import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym


class Scorer:
    NWOT = 'nwot'
    ACC = 'acc'
    LOSS = 'loss'

    def __init__(self, model:nn.Module, function:str):
        self.model = model
        self.function = function

        if function == Scorer.NWOT:
            self.score = self._nwot
        elif function == Scorer.ACC:
            self.score = self._acc
        elif function == Scorer.LOSS:
            self.score = self._loss
        else:
            raise ValueError('Invalid function')
    
    def __call__(self, x, y, get_logits=False):
        mode = self.model.training
        self.model.eval()

        metric, logits = self.score(x, y)

        self.model.train(mode)
        
        if get_logits:
            return metric, logits
        else:
            return metric
    
    @torch.no_grad()
    def _nwot(self, x, y):
        batch_size = x.size(0)
        self.K = torch.zeros(batch_size, batch_size).to(x.device)
        hooks = []
        for _, m in self.model.named_modules():
            if isinstance(m, nn.ReLU):
                hooks.append(m.register_forward_hook(self._forward_hook))

        logits = self.model(x)
        K = self.K.cpu().numpy()
        s, ld = np.linalg.slogdet(K)

        for h in hooks:
            h.remove()
    
        return ld.item(), logits
    
    @torch.no_grad()
    def  _acc(self, x, y):
        logits = self.model(x)
        preds = torch.argmax(logits, dim=-1)
        return (preds == y).float().mean().item(), logits
    
    @torch.no_grad()
    def _loss(self, x, y):
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        # We want to minimize the loss, so we return the negative loss.
        return -loss.item(), logits

    def _forward_hook(self, module, inp, out):
        if isinstance(inp, tuple):
            inp = inp[0]
        batch_size = inp.size(0)
        inp = inp.view(batch_size, -1)
        x = (inp > 0).float()
        K = x @ x.t()
        K2 = (1 - x) @ (1 - x.t())
        self.K = self.K + K + K2


class DiscreteDartsRL(gym.Env):
    def __init__(self, model, data_loader, scoring_fn=Scorer.NWOT, device='cuda'):
        self.model = model
        self.data_loader = data_loader
        self.scoring_fn = scoring_fn
        self.device = device
        # Each cell has 14 total edges. We optimize each edge. There are 2 cells (normal and reduce)
        self.max_steps = sum(range(2, 6)) * 2
        self.scorer = Scorer(model, scoring_fn)

        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(model.alpha_normal.numel() + model.alpha_reduce.numel(), )
        )
        # The action picks the operation for each edge.
        self.action_space = gym.spaces.Discrete(len(model.primitives))

        self._total_steps = 0

    def reset(self, seed = None, options = None):
        super().reset(seed=seed, options=options)
        self.model.reset_alphas()
        self._step = 0
        return self._get_obs(), {}

    def step(self, action, sentinel=1e6):
        # 1. Pick between normal and reduce cell.
        alpha = self.model.alphas[self._step >= self.max_steps // 2]
        # 2. Pick the correct edge based on step.
        edge = alpha[self._step % alpha.size(0)]

        with torch.no_grad():
            edge[action] = sentinel
        
        self._step += 1
        self._total_steps += 1

        if self._is_done() and self._total_steps % (self.max_steps * 100) == 0:
            logging.info(f"Total steps: {self._total_steps}, Genotype: {self.model.genotype}")

        return self._get_obs(), self._get_reward(), self._is_done(), False, {}
    
    def _get_obs(self):
        obs = torch.stack([F.softmax(alpha, dim=-1) for alpha in self.model.alphas])
        return obs.flatten().detach().cpu().numpy()

    def _get_reward(self):
        def _next():
            try:
                x, y = next(self._iter)
            except (StopIteration, AttributeError) as e:
                self._iter = iter(self.data_loader)
                x, y = next(self._iter)
            
            if x.shape[0] == self.data_loader.batch_size:
                return x.to(self.device), y.to(self.device)
            else:
                return _next()

        if self._is_done():
            return self.scorer(*_next())
        else:
            return 0

    def _is_done(self):
        return self._step >= self.max_steps
    

class ContinuousDartsRL(gym.Env):
    def __init__(self, model, train_loader, valid_loader, iters=80, scoring_fn='acc', grad=False, grad_step=False, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.scoring_fn = scoring_fn
        self.grad = grad
        self.grad_step = grad_step
        self.device = device
        # Each cell has 14 total edges. Each edge is optimized `iters` times. There are 2 cells (normal and reduce)
        self.max_steps_per_iter = sum(range(2, 6)) * 2
        self.max_steps = self.max_steps_per_iter * iters

        obs_space = model.alpha_normal.numel() + model.alpha_reduce.numel()
        if grad: obs_space *= 2
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_space, ))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(len(model.primitives), ))

        if grad_step:
            self.optimizer = torch.optim.SGD(model.weights, lr=0.001, momentum=0.9, weight_decay=3e-4)
        self._total_steps = 0

    def reset(self, seed = None, options = None):
        super().reset(seed=seed, options=options)
        self.model.reset_alphas()
        self._step = 0
        return self._get_obs(), {}
    
    def step(self, action, lr=0.01):
        iter_step = self._step % self.max_steps_per_iter
        # 1. Pick between normal and reduce cell.
        alpha = self.model.alphas[iter_step >= self.max_steps_per_iter // 2]
        # 2. Pick the correct edge based on step.
        edge = alpha[iter_step % alpha.size(0)]

        action = torch.tensor(action, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            edge += lr * action.reshape_as(edge)

        # If gradient stepping is enabled, take a step at end of each iteration.
        if self.grad_step and iter_step == self.max_steps_per_iter - 1:
            self._model_step(train=True)
        
        self._step += 1
        self._total_steps += 1

        # Forward pass is needed only if `grads` is enabled or the reward needs to be calculated.
        if self.grad:
            loss, acc = self._model_step(train=False)
            obs = self._get_obs(loss)
            if self._is_done():
                reward = -loss if self.scoring_fn == 'loss' else acc
            else:
                reward = 0
        else:
            obs = self._get_obs()
            if self._is_done():
                loss, acc = self._model_step(train=False)
                reward = -loss if self.scoring_fn == 'loss' else acc
            else:
                reward = 0

        if self._is_done() and self._total_steps % (self.max_steps * 100) == 0:
            logging.info(f"Total steps: {self._total_steps}, Genotype: {self.model.genotype}")
        
        return obs, reward, self._is_done(), False, {}

    def _get_obs(self, loss=None):
        alpha_normal = F.softmax(self.model.alpha_normal, dim=-1)
        alpha_reduce = F.softmax(self.model.alpha_reduce, dim=-1)

        if not self.grad:
            obs = torch.stack([alpha_normal, alpha_reduce])
            return obs.flatten().detach().cpu().numpy()

        if loss is None:
            d_alpha_normal = torch.zeros_like(alpha_normal)
            d_alpha_reduce = torch.zeros_like(alpha_reduce)
        else:
            d_alpha_normal, d_alpha_reduce = torch.autograd.grad(
                loss, [self.model.alpha_normal, self.model.alpha_reduce]
            )
        
        obs = torch.stack([alpha_normal, d_alpha_normal, alpha_reduce, d_alpha_reduce])
        return obs.flatten().detach().cpu().numpy()

    def _is_done(self):
        return self._step >= self.max_steps

    def _model_step(self, train=True):
        mode = self.model.training
        self.model.train(train)
        x, y = self._t_next() if train else self._v_next()

        logits = self.model(x)
        loss = F.cross_entropy(logits, y)

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


        self.model.train(mode=mode)
        return loss, (logits.argmax(dim=-1) == y).float().mean().item()

    def _t_next(self):
        try:
            x, y = next(self.r_iter)
        except (StopIteration, AttributeError) as e:
            self.r_iter = iter(self.train_loader)
            x, y = next(self.r_iter)
        
        if x.shape[0] == self.train_loader.batch_size:
            return x.to(self.device), y.to(self.device)
        else:
            return self._t_next()

    def _v_next(self):
        try:
            x, y = next(self.v_iter)
        except (StopIteration, AttributeError) as e:
            self.v_iter = iter(self.valid_loader)
            x, y = next(self.v_iter)
        
        if x.shape[0] == self.valid_loader.batch_size:
            return x.to(self.device), y.to(self.device)
        else:
            return self._v_next()
        