#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm

attacks_data = {
    "correlated_attack": np.array([
        [0.36, 0.01, 0.36, 0.33, 0.39],
        [2.73, 0.06, 2.72, 2.62, 2.90],
        [3.73, 0.11, 3.74, 3.44, 4.02],
        [10.45, 1.12, 10.38, 8.14, 15.67]
    ]),
    "max_engine_attack": np.array([
        [0.35, 0.02, 0.35, 0.03, 0.46],
        [2.61, 0.08, 2.60, 2.42, 2.94],
        [4.51, 0.13, 4.51, 4.16, 4.86],
        [10.16, 1.67, 9.98, 8.18, 21.18]
    ]),
    "max_speedometer_attack": np.array([
        [0.35, 0.01, 0.35, 0.34, 0.39],
        [2.53, 0.03, 2.53, 2.48, 2.66],
        [3.58, 0.04, 3.58, 3.45, 3.71],
        [10.45, 0.62, 10.48, 8.88, 13.33]
    ]),
    "light_off_attack": np.array([
        [0.35, 0.01, 0.35, 0.33, 0.38],
        [2.63, 0.04, 2.63, 2.54, 2.82],
        [3.79, 0.08, 3.79, 3.59, 4.00],
        [9.27, 0.81, 9.19, 7.87, 13.72]
    ]),
    "light_on_attack": np.array([
        [0.35, 0.01, 0.35, 0.33, 0.37],
        [2.63, 0.05, 2.69, 2.57, 2.86],
        [3.79, 0.08, 3.85, 3.62, 4.02],
        [9.27, 0.65, 11.09, 9.03, 12.81]
    ])
}

labels = np.array([r'$\mu$', r'$\sigma$', r'$\eta$', 'min', 'max'])
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  

# Method names
legend_labels = [
    "Matrix Correlation Distribution",
    "Matrix Correlation Correlation",
    "Ganesan17",
    "Moriano22"
]
num_methods = len(legend_labels)
colors_inferno = cm.inferno(np.linspace(0.55, 1, num_methods))
custom_lines_inferno = [Line2D([0], [0], color=colors_inferno[i], lw=4) for i in range(num_methods)]

fig, axs = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(polar=True))
fig.subplots_adjust(hspace=0.3, wspace=0, top=0.87, bottom=0.14)

for i, (attack_name, data) in enumerate(attacks_data.items()):
    stats = np.concatenate((data, data[:,[0]]), axis=1)
    ax = axs.flatten()[i]
    for j, stat in enumerate(stats):
        ax.plot(angles, stat, label=legend_labels[j], color=colors_inferno[j])
        ax.fill(angles, stat, color=colors_inferno[j], alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=12)

fig.delaxes(axs.flatten()[-1])

legend = fig.legend(custom_lines_inferno, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=4)
plt.setp(legend.get_title(), fontsize='16')
plt.savefig(f'/Users/6u0/Desktop/Article Submissions/Paper3/polar_plot_min.pdf', format='pdf', bbox_inches='tight')

plt.show()

