import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def add_labels(ax,x, y):
    for i, j in zip(x, y):
        ax.text(i, j, round(j, 2), ha = 'center')
        
def multi_bar_plots(data, labels, x_tick_labels, x_label, y_label,
                     titles, sup_title, yerr=False, figsize=(10, 4)):
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    for i, title in enumerate(titles):
        plot_multi_bar(data[i], labels=labels, x_tick_labels=x_tick_labels,
                       x_label=x_label, y_label=y_label, title=title,
                       yerr=yerr, ax=ax[i])
    plt.suptitle(sup_title)
    plt.show()

def plot_multi_bar(data, labels, x_tick_labels, x_label, y_label,
                title, yerr = False, ax = None):
    
    n = data.shape[1]
    width = 0.8 / n
    show_plot = ax == None
    if ax is None:
        fig, ax = plt.subplots()

    x_start_pos = np.arange(-(n-1)/2, (n+1)/2)
    for i in range(n):
        y_i = data[:, i]
        m = x_start_pos[i]
        x = np.arange(y_i.shape[0])
        label = labels[i]
        if yerr:
            mu, std = y_i[:,0], y_i[:,1]
            ax.bar(x + m * width, mu, yerr = std, width = width, label = label,
                alpha=0.5, ecolor='black', capsize=5)
            add_labels(ax, x + m * width, mu)
        else:
            ax.bar(x + m * width, y_i, width = width, label = label,
                alpha=0.5)
            add_labels(ax, x + m * width, y_i)
    
    ax.set_xticks(x)
    ax.set_xticklabels(x_tick_labels)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.set_title(title)
    show_plot and plt.show()
