import numpy as np
import torch
import random
import torch.nn.functional as F
from utils.dreamer_style_loss import symlog, symexp


def act_metric(preds, gts, prefix='val', start_stop_interval=[(0,1),(1,9),(9,25),(25,57)]):
    """
    inputs:
        preds    : b, t, nc_act
        gts      : b, t, nc_act
        start_stop_interval: how to split action predictions along the temporal dimension, like [(0, t-1), (t-1, t)]
    outputs:
        MSE of actions
    """
    assert preds.shape == gts.shape
    assert start_stop_interval[0][0] == 0 and start_stop_interval[-1][-1] == preds.shape[1]
    logs = {}
    for i in range(preds.shape[-1]):
        dim_delta = (preds[:,:,i] - gts[:,:,i]) ** 2
        dim_mean = dim_delta.mean(axis=0)
        dim_std = dim_delta.std(axis=0)
        for h_start, h_stop in start_stop_interval:
            logs[f'{prefix}/{h_start}_{h_stop}_dim_{i}_diff'] = np.mean(dim_mean[h_start:h_stop])
            logs[f'{prefix}/{h_start}_{h_stop}_dim_{i}_std'] = np.mean(dim_std[h_start:h_stop])

    return logs


def value_metric(preds, gts, prefix='val', start_stop_interval=[(0,1),(1,9),(9,25),(25,57)], save_plot=False, plot_dir=None, bins = None):
    """
    inputs:
        preds    : b, t, num_bins (twohot encoded logits or probabilities)
        gts      : b, t, c (continous values - only the last dim is the actual value)
        bins     : (num_bins,) - the bins used for twohot encoding
        start_stop_interval: how to split value predictions along the temporal dimension
        save_plot: whether to save value comparison plot
        plot_dir: directory to save the plot
    outputs:
        MSE and other metrics of values (after converting from twohot to continuous)
    """
    assert start_stop_interval[0][0] == 0 and start_stop_interval[-1][-1] == preds.shape[1]
    logs = {}

    # Convert twohot back to continuous values using weighted average with bins
    def twohot_to_continuous(twohot_logits, bins):
        soft_p = F.softmax(twohot_logits, dim=-1)
        continuous = soft_p @ bins
        return continuous

    if bins is not None:
        preds_continuous = twohot_to_continuous(preds, bins)
    else:
        preds_continuous = symexp(preds)
        preds_continuous = preds_continuous[..., -1]  # shape: b, t

    if gts.dim() == 3:
        gts_continuous = gts[..., -1]  # shape: b, t
        symlog_gts = symlog(gts_continuous)  # Convert to log space if needed
    else:
        gts_continuous = gts  # shape: b, t

    # Compute MSE
    mse = (preds_continuous - gts_continuous) ** 2
    mse_mean = mse.mean(axis=0)
    mse_std = mse.std(axis=0)

    for h_start, h_stop in start_stop_interval:
        logs[f'{prefix}/{h_start}_{h_stop}_value_mse'] = np.mean(mse_mean[h_start:h_stop].detach().cpu().numpy())
        logs[f'{prefix}/{h_start}_{h_stop}_value_std'] = np.mean(mse_std.detach().cpu().numpy())

    # Save value comparison plot if requested
    if save_plot and plot_dir is not None:
        import matplotlib.pyplot as plt
        import os

        os.makedirs(plot_dir, exist_ok=True)

        b, t = preds_continuous.shape
        x_axis = range(t)

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Plot the continuous value sequences for the first sample in batch
        ax.plot(x_axis, gts_continuous[0].detach().cpu().numpy(), label='Ground Truth', color='cornflowerblue', alpha=0.9)
        ax.plot(x_axis, preds_continuous[0].detach().cpu().numpy(), label='Inferred', color='tomato', linestyle='--', alpha=0.9)

        # Mark the starting point
        ax.scatter([0], gts_continuous[0, 0].detach().cpu().numpy(), c='blue', marker='o', s=40, zorder=5, label='GT Start')
        ax.scatter([0], preds_continuous[0, 0].detach().cpu().numpy(), c='darkred', marker='x', s=40, zorder=5, label='Inferred Start')

        ax.set_title(f'Value Prediction Comparison - {prefix}')
        ax.set_ylabel('Value')
        ax.set_xlabel('Timestep')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()

        fig.suptitle(f'Value Prediction: {prefix}', fontsize=18)

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plot_path = os.path.join(plot_dir, f'value_comparison_{prefix}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.clf()

        print(f"Saved value comparison plot to {plot_path}")

        # Save comparison plot for symlog_gts vs preds[..., -1]
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Plot symlog_gts vs preds[..., -1] for the first sample in batch
        ax.plot(x_axis, symlog_gts[0].detach().cpu().numpy(), label='symlog_gts', color='forestgreen', alpha=0.9)
        ax.plot(x_axis, preds[0, :, -1].detach().cpu().numpy(), label='preds', color='orange', linestyle='--', alpha=0.9)

        # Mark the starting point
        ax.scatter([0], symlog_gts[0, 0].detach().cpu().numpy(), c='darkgreen', marker='o', s=40, zorder=5, label='symlog_gts Start')
        ax.scatter([0], preds[0, 0, -1].detach().cpu().numpy(), c='darkorange', marker='x', s=40, zorder=5, label='Inferred Start')

        ax.set_title(f'symlog_gts vs Inferred Start Comparison - {prefix}')
        ax.set_ylabel('Value')
        ax.set_xlabel('Timestep')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()

        fig.suptitle(f'symlog_gts vs preds: {prefix}', fontsize=18)

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plot_path = os.path.join(plot_dir, f'symlog_vs_preds_{prefix}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.clf()

        print(f"Saved symlog vs preds_last comparison plot to {plot_path}")

    return logs