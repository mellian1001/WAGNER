import pathlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import z_temp.common as common


def plot_counts(
    inpath, outpath, color, budget=1e5, cols=4, size=(2, 1.8)):
  runs = common.load_runs([inpath], budget)
  percents, methods, seeds, tasks = common.compute_success_rates(runs, budget)
  borders = np.arange(0, budget, 1e3)
  keys = ['reward', 'length'] + tasks

  rows = len(keys) // cols
  fig, axes = plt.subplots(
      nrows=rows, ncols=cols, figsize=(size[0] * cols, size[1] * rows))

  for ax, key in zip(axes.flatten(), keys):
    ax.set_title(key.replace('achievement_', '').replace('_', ' ').title())
    xs = np.concatenate([run['xs'] for run in runs])
    ys = np.concatenate([run[key] for run in runs])

    binxs, binys = common.binning(xs, ys, borders, np.nanmean)
    ax.plot(binxs, binys, color=color)

    mins = common.binning(xs, ys, borders, np.nanmin)[1]
    maxs = common.binning(xs, ys, borders, np.nanmax)[1]
    ax.fill_between(binxs, mins, maxs, linewidths=0, alpha=0.2, color=color)

    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))

    if maxs.max() == 0:
      ax.set_ylim(-0.05, 1.05)

  fig.tight_layout()
  pathlib.Path(outpath).parent.mkdir(exist_ok=True, parents=True)
  fig.savefig(outpath)
  print(f'Saved {outpath}')


plot_counts(
    'scores/crafter_reward-our_method.json',
    'plots/counts-our_method.pdf', '#377eb8')
