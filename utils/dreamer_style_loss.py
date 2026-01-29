import torch
import torch.nn.functional as F


def symlog(x):
    return torch.sign(x) * torch.log1p(x.abs())

def symexp(x):
    return torch.sign(x) * (torch.exp(x.abs()) - 1)


def build_twohot_bins(num_bins=41, min_exp=-5, max_exp=5):
    exps = torch.linspace(min_exp, max_exp, num_bins)
    bins = symexp(exps)
    return bins   # (num_bins,)


def twohot_encode(y, bins):
    """
    y:    (batch,)
    bins: (N,)
    输出: twohot (batch, N)
    """
    batch = y.shape[0]
    N = bins.shape[0]
    twohot = torch.zeros(batch, N, device=y.device)
    bins = bins.to(y.device)
    # 对每个样本找到最接近的两个邻居
    for i in range(batch):
        yi = y[i]
        # searchsorted 找到 yi 落在哪个区间
        idx = torch.searchsorted(bins, yi).item() - 1
        idx = max(0, min(idx, N - 2))   # ↓ 保证合法

        b0 = bins[idx]
        b1 = bins[idx + 1]

        d = b1 - b0
        w0 = (b1 - yi) / d
        w1 = (yi - b0) / d

        twohot[i, idx]     = w0
        twohot[i, idx + 1] = w1

    return twohot


# =================================================================
# 4. Dreamer-style reward/value predictor loss（twohot categorical CE）
# =================================================================
def dreamer_twohot_loss(logits, targets_twohot):
    """
    logits: (batch, N)  —— 网络输出，未softmax
    targets_twohot: (batch, N)
    """
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(targets_twohot * log_probs).sum(dim=1).mean()
    return loss


