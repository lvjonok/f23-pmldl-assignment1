"""Utils module in visualization package provides wrappers for matplotlib"""

import matplotlib.pyplot as plt


def simple_plot(data, title, xlabel, ylabel, save_path=None, **kwargs):
    """simple_plot plots a simple line graph"""
    plt.plot(data, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    if save_path:
        plt.savefig(save_path)


def simple_hist(data, title, xlabel, ylabel, save_path=None, **kwargs):
    """simple_hist plots a simple histogram"""
    plt.hist(data, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()
    if save_path:
        plt.savefig(save_path)


def multiple_hist(datas, title, xlabel, ylabel, save_path=None, **kwargs):
    """multiple_hist plots a simple histogram"""
    for data in datas:
        plt.hist(data, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    if save_path:
        plt.savefig(save_path)
