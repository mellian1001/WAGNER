import pathlib

import numpy as np
import matplotlib.pyplot as plt

import z_temp.common as common


def plot_scores(inpaths, outpath, legend, colors, budget=1e4, ylim=None):
  runs = common.load_runs(inpaths, budget)
  percents, methods, seeds, tasks = common.compute_success_rates(runs, budget)
  print(percents)
  scores = common.compute_scores(percents)
  if not legend:
    methods = sorted(set(run['method'] for run in runs))
    legend = {x: x.replace('_', ' ').title() for x in methods}
  legend = dict(reversed(legend.items()))

  scores = scores[np.array([methods.index(m) for m in legend.keys()])]
  mean = np.nanmean(scores, -1)
  std = np.nanstd(scores, -1)

  fig, ax = plt.subplots(figsize=(4, 3))
  centers = np.arange(len(legend))
  width = 0.7
  colors = list(reversed(colors[:len(legend)]))
  error_kw = dict(capsize=5, c='#000')
  ax.bar(centers, mean, yerr=std, color=colors, error_kw=error_kw)

  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.tick_params(
      axis='x', which='both', width=50, length=0.8, direction='inout')
  ax.set_xlim(centers[0] - 2 * (1 - width), centers[-1] + 2 * (1 - width))
  ax.set_xticks(centers + 0.0)
  ax.set_xticklabels(
      list(legend.values()), rotation=45, ha='right', rotation_mode='anchor')

  ax.set_ylabel('Crafter Score (%)')
  if ylim:
    ax.set_ylim(0, ylim)

  fig.tight_layout()
  pathlib.Path(outpath).parent.mkdir(exist_ok=True, parents=True)
  fig.savefig(outpath)
  print(f'Saved {outpath}')


inpaths = [
    'scores/crafter_reward-base.json',
    'scores/crafter_reward-our_method.json',
    'scores/crafter_reward-LLM_only.json',
]
legend = {
    'base': 'Base',
    'our_method': 'Our Method',
    'LLM_only': "LLM Only",
}
colors = [
    '#cccccc',
    '#377eb8', '#5fc35d', '#984ea3',
    '#bf3217', '#de9f42', '#6a554d',
]
plot_scores(inpaths, 'plots/scores.pdf', legend, colors, ylim=20)

# inpaths = [
#     'scores/crafter_reward-dreamerv2.json',
#     'scores/crafter_reward-ppo.json',
#     'scores/crafter_reward-rainbow.json',
#     'scores/crafter_noreward-unsup_plan2explore.json',
#     'scores/crafter_noreward-unsup_rnd.json',
#     'scores/crafter_noreward-random.json',
# ]
# legend = {
#     'dreamerv2': 'DreamerV2',
#     'ppo': 'PPO',
#     'rainbow': 'Rainbow',
#     'unsup_plan2explore': 'Plan2Explore\n(Unsup)',
#     'unsup_rnd': 'RND\n(Unsup)',
#     'random': 'Random',
# }
# colors = [
#     '#377eb8', '#5fc35d', '#984ea3',
#     '#bf3217', '#de9f42', '#6a554d',
# ]
# plot_scores(inpaths, 'plots/scores-agents.pdf', legend, colors, ylim=12)
