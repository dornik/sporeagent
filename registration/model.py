import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np

from config import *


class Agent(nn.Module):

    def __init__(self, num_classes=21):
        super().__init__()

        self.num_classes = num_classes  # for segmentation/outlier removal
        self.state_emb = StateEmbed(num_classes)
        self.actor_critic = ActorCriticHead()

    def forward(self, src, tgt, tgt_emb=None, class_idx=None):
        # -- EMBEDDING
        # O(src, tgt) -> S
        state, emb_tgt, est_seg_src = self.state_emb(src, tgt, tgt_emb, class_idx)

        # S -> a, v
        action, value = self.actor_critic(state)

        # -- ACTION/VALUE HEAD
        # reshape a to B x axis x [step, sign]
        action = (action[0].view(-1, 3, 2 * NUM_STEPSIZES + 1),
                  action[1].view(-1, 3, 2 * NUM_STEPSIZES + 1))
        value = value.view(-1, 1, 1)

        return state, action, value, emb_tgt, est_seg_src


class StateEmbed(nn.Module):

    def __init__(self, num_classes=21):
        super().__init__()

        conv_dims = [64, 128, 256, 512, 1024]
        self.conv1 = nn.Conv1d(IN_CHANNELS, conv_dims[0], 1)
        self.conv2 = nn.Conv1d(conv_dims[0], conv_dims[1], 1)
        self.conv2a = nn.Conv1d(conv_dims[1], conv_dims[2], 1)
        self.conv2b = nn.Conv1d(conv_dims[2], conv_dims[3], 1)
        self.conv3 = nn.Conv1d(conv_dims[3], conv_dims[-1], 1)

        if USE_SEGMENT:
            self.segment = nn.Sequential(
                nn.Conv1d(conv_dims[0]+conv_dims[-1]+num_classes, 512, 1),
                nn.ReLU(),
                nn.Conv1d(512, 256, 1),
                nn.ReLU(),
                nn.Conv1d(256, 128, 1),
                nn.ReLU(),
                nn.Conv1d(128, 1, 1)
            )

    def forward(self, src, tgt, tgt_emb=None, class_idx=None):
        B, N, D = src.shape

        # O=(src,tgt) -> S=[Phi(src), Phi(tgt)]
        if USE_SEGMENT:
            # Bx1088xN, Bx1024, Bx1024xN
            emb_point_src, emb_src, emb_prepool_src = self.embed(src.transpose(2, 1), per_point=True)
            # inspired by Frustum PointNets (add one-hot instance label)
            emb_point_src = torch.cat([class_idx[..., None].repeat(1, 1, N), emb_point_src], dim=1)
            emb_seg_src = self.segment(emb_point_src)  # Bx1xN
            est_seg_src = torch.sigmoid(emb_seg_src.transpose(2, 1))  # use BCELoss

            # pool over inliers
            mask = est_seg_src.transpose(2, 1)
            emb_masked_prepool = torch.where(mask > 0.5,
                                             emb_prepool_src, torch.ones_like(mask) * emb_prepool_src.min())
            emb_src = torch.max(emb_masked_prepool, 2, keepdim=True)[0].view(-1, 1024)
        else:
            emb_src = self.embed(src.transpose(2, 1))
            est_seg_src = torch.ones_like(src[..., 0][..., None])  # Bx1xN, all inliers
        emb_tgt = self.embed(tgt.transpose(2, 1))
        state = torch.cat((emb_src, emb_tgt), dim=-1)
        state = state.view(B, -1)

        return state, emb_tgt, est_seg_src

    def embed(self, x, per_point=False):
        B, D, N = x.shape

        # embedding: BxDxN -> BxFxN
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x2a = F.relu(self.conv2a(x2))
        x2b = F.relu(self.conv2b(x2a))
        x3 = self.conv3(x2b)

        # pooling: BxFxN -> BxFx1
        x_pooled = torch.max(x3, 2, keepdim=True)[0]

        if per_point:
            # propagation of global embedding: BxFgxN -> Bx(Fg+Fl)xN
            x_perpoint = x_pooled.repeat(1, 1, N)
            x_perpoint = torch.cat([x_perpoint, x1], 1)
            return x_perpoint, x_pooled.view(B, -1), x3  # Bx(Fg+Fl)xN and BxFg (global embedding for later pooling)
        else:
            return x_pooled.view(B, -1)  # global feature BxFg


class ActorCriticHead(nn.Module):

    def __init__(self):
        super().__init__()

        self.activation = nn.ReLU()

        self.emb_r = nn.Sequential(
            nn.Linear(STATE_DIM, HEAD_DIM*2),
            self.activation,
            nn.Linear(HEAD_DIM*2, HEAD_DIM),
            self.activation
        )
        self.action_r = nn.Linear(HEAD_DIM, NUM_ACTIONS * NUM_STEPSIZES + NUM_NOPS)

        self.emb_t = nn.Sequential(
            nn.Linear(STATE_DIM, HEAD_DIM*2),
            self.activation,
            nn.Linear(HEAD_DIM*2, HEAD_DIM),
            self.activation
        )
        self.action_t = nn.Linear(HEAD_DIM, NUM_ACTIONS * NUM_STEPSIZES + NUM_NOPS)

        self.emb_v = nn.Sequential(
            nn.Linear(HEAD_DIM * 2, HEAD_DIM),
            self.activation
        )
        self.value = nn.Linear(HEAD_DIM, 1)

    def forward(self, state):
        # S -> S'
        emb_t = self.emb_t(state)
        emb_r = self.emb_r(state)
        # S' -> pi
        action_logits_t = self.action_t(emb_t)
        action_logits_r = self.action_r(emb_r)

        # S' -> v
        state_action = torch.cat([emb_t, emb_r], dim=1)
        emb_v = self.emb_v(state_action)
        value = self.value(emb_v)

        return [action_logits_t, action_logits_r], value


# -- action helpers
def action_from_logits(logits, deterministic=True):
    distributions = _get_distributions(*logits)
    actions = _get_actions(*(distributions + (deterministic,)))

    return torch.stack(actions).transpose(1, 0)


def action_stats(logits, action):
    distributions = _get_distributions(*logits)
    logprobs, entropies = _get_logprob_entropy(*(distributions + (action[:, 0], action[:, 1])))

    return torch.stack(logprobs).transpose(1, 0), torch.stack(entropies).transpose(1, 0)


def _get_distributions(action_logits_t, action_logits_r):
    distribution_t = Categorical(logits=action_logits_t)
    distribution_r = Categorical(logits=action_logits_r)

    return distribution_t, distribution_r


def _get_actions(distribution_t, distribution_r, deterministic=True):
    if deterministic:
        action_t = torch.argmax(distribution_t.probs, dim=-1)
        action_r = torch.argmax(distribution_r.probs, dim=-1)
    else:
        action_t = distribution_t.sample()
        action_r = distribution_r.sample()
    return action_t, action_r


def _get_logprob_entropy(distribution_t, distribution_r, action_t, action_r):
    logprob_t = distribution_t.log_prob(action_t)
    logprob_r = distribution_r.log_prob(action_r)

    entropy_t = distribution_t.entropy()
    entropy_r = distribution_r.entropy()

    return [logprob_t, logprob_r], [entropy_t, entropy_r]


# --- model helpers
def load(model, path):
    infos = torch.load(path)
    model.load_state_dict(infos['model_state_dict'])
    return infos


def save(model, path, infos={}):
    infos['model_state_dict'] = model.state_dict()
    torch.save(infos, path)


def plot_grad_flow(model):
    """
    via https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7

    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ave_grads = []
    max_grads = []
    layers = []
    for n, p in model.named_parameters():
        if (p.requires_grad) and ("bias" not in n):
            if p.grad is None:
                print(f"no grad for {n}")
                continue
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, -1, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=-1, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=torch.max(torch.stack(max_grads)).cpu())
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
