#!/usr/bin/env python
# coding: utf-8

# correlated_attack:
# 
# Mean ($\mu$): [0.57, 0.65, 0.49, 0.69]
# Standard Deviation ($\sigma$): [0.05, 0.06, 0.07, 0.10]
# Median ($\eta$): [0.57, 0.65, 0.49, 0.71]
# Min: [0.44, 0.52, 0.31, 0.27]
# Max: [0.81, 0.52, 0.31, 0.87]
# 
# max_engine_attack:
# 
# Mean ($\mu$): [0.34, 0.49, 0.31, 0.67]
# Standard Deviation ($\sigma$): [0.11, 0.11, 0.11, 0.11]
# Median ($\eta$): [0.33, 0.47, 0.29, 0.67]
# Min: [0.09, 0.24, 0.04, 0.29]
# Max: [0.68, 0.87, 0.63, 0.93]
# 
# max_speedometer_attack:
# 
# Mean ($\mu$): [0.43, 0.33, 0.65, 0.28]
# Standard Deviation ($\sigma$): [0.03, 0.02, 0.03, 0.03]
# Median ($\eta$): [0.43, 0.33, 0.64, 0.28]
# Min: [0.26, 0.25, 0.56, 0.17]
# Max: [0.56, 0.39, 0.74, 0.45]
# 
# light_off_attack:
# 
# Mean ($\mu$): [0.69, 0.53, 0.33, 0.54]
# Standard Deviation ($\sigma$): [0.04, 0.05, 0.04, 0.05]
# Median ($\eta$): [0.71, 0.53, 0.33, 0.55]
# Min: [0.55, 0.36, 0.24, 0.36]
# Max: [0.79, 0.64, 0.45, 0.73]
# 
# light_on_attack:
# 
# Mean ($\mu$): [0.43, 0.49, 0.49, 0.45]
# Standard Deviation ($\sigma$): [0.03, 0.03, 0.03, 0.03]
# Median ($\eta$): [0.43, 0.49, 0.49, 0.45]
# Min: [0.29, 0.41, 0.42, 0.35]
# Max: [0.53, 0.60, 0.59, 0.58]

# In[12]:


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.lines import Line2D

# Data for all attacks
attacks_data = {
    "correlated_attack": np.array([
        [0.57, 0.05, 0.57, 0.44, 0.81],
        [0.65, 0.06, 0.65, 0.52, 0.52],
        [0.49, 0.07, 0.49, 0.31, 0.31],
        [0.69, 0.10, 0.71, 0.27, 0.87]
    ]),
    "max_engine_attack": np.array([
        [0.34, 0.11, 0.33, 0.09, 0.68],
        [0.49, 0.11, 0.47, 0.24, 0.87],
        [0.31, 0.11, 0.29, 0.04, 0.63],
        [0.67, 0.11, 0.67, 0.29, 0.93]
    ]),
    "max_speedometer_attack": np.array([
        [0.43, 0.03, 0.43, 0.26, 0.56],
        [0.33, 0.02, 0.33, 0.25, 0.39],
        [0.65, 0.03, 0.64, 0.56, 0.74],
        [0.28, 0.03, 0.28, 0.17, 0.45]
    ]),
    "light_off_attack": np.array([
        [0.69, 0.04, 0.71, 0.55, 0.79],
        [0.53, 0.05, 0.53, 0.36, 0.64],
        [0.33, 0.04, 0.33, 0.24, 0.45],
        [0.54, 0.05, 0.55, 0.36, 0.73]
    ]),
    "light_on_attack": np.array([
        [0.43, 0.03, 0.43, 0.29, 0.53],
        [0.49, 0.03, 0.49, 0.41, 0.60],
        [0.49, 0.03, 0.49, 0.42, 0.59],
        [0.45, 0.03, 0.45, 0.35, 0.58]
    ])
}

labels = np.array([r'$\mu$', r'$\sigma$', r'$\eta$', 'min', 'max'])
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Complete the loop

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

# Plotting
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

# Remove the last subplot
fig.delaxes(axs.flatten()[-1])

legend = fig.legend(custom_lines_inferno, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=4)
plt.setp(legend.get_title(), fontsize='16')
plt.show()

