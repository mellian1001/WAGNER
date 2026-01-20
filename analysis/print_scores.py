import numpy as np

import z_temp.common as common


def print_scores(inpaths, legend, budget=1e6, sort=False):
  runs = common.load_runs(inpaths, budget)
  percents, methods, seeds, tasks = common.compute_success_rates(runs, budget)
  scores = common.compute_scores(percents)
  if not legend:
    methods = sorted(set(run['method'] for run in runs))
    legend = {x: x.replace('_', ' ').title() for x in methods}

  scores = scores[np.array([methods.index(m) for m in legend.keys()])]
  # print(scores)
  means = np.nanmean(scores, -1)
  stds = np.nanstd(scores, -1)

  print('')
  print(r'\textbf{Method} & \textbf{Score} \\')
  print('')
  for method, mean, std in zip(legend.values(), means, stds):
    mean = f'{mean:.1f}'
    mean = (r'\o' if len(mean) < 4 else ' ') + mean
    print(rf'{method:<25} & ${mean} \pm {std:4.1f}\%$ \\')
  print('')


inpaths = [
    'scores/crafter_reward-base.json',
    'scores/crafter_reward-our_method.json',
    'scores/crafter_reward-LLM_only.json',
]
legend = {
    'base': 'Base',
    'our_method': "Our Method",
    'LLM_only': 'LLM Only',
}
print_scores(inpaths, legend)
