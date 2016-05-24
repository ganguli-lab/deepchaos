import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def get_pal(ncolors, npercolor, plot=False):
    base_colors = sns.color_palette("deep")[:ncolors]
    n_off = npercolor // 3.
    pal = np.vstack([sns.light_palette(c, npercolor+n_off, reverse=True)[:npercolor] for c in base_colors])
    sns.set_palette(pal)
    if plot:
        sns.palplot(pal)
    return pal

def add_label(label, xoff=-0.1, yoff=1.3):
    ax = plt.gca()
    ax.text(xoff, yoff, '%s'%label, transform=ax.transAxes,
      fontsize=12, fontweight='bold', va='top', ha='right')

def pcolor(*args, **kwargs):
    """Version of pcolor that removes edges""" 
    h = plt.pcolormesh(*args, **kwargs)
    h.set_edgecolor('face')
    return h


def sigma_pcolor(q, weight_sigmas, bias_sigmas, draw_colorbar=True, **kwargs):
    if 'vmax' not in kwargs:
        kwargs['vmax'] = int(np.ceil(np.nanmax(q)))
    pcolor(bias_sigmas, weight_sigmas, q, cmap=plt.cm.viridis, vmin=0, **kwargs)
    plt.yticks(weight_sigmas[weight_sigmas == weight_sigmas.astype(int)])
    plt.xticks(bias_sigmas[bias_sigmas == bias_sigmas.astype(int)])
    plt.xlabel('$\sigma_b$')
    plt.ylabel('$\sigma_w$')
    cmax = kwargs['vmax']
    if draw_colorbar:
        plt.colorbar(ticks=(0, cmax/2.0, cmax))

