import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_probability as tfp


def save2Vis(figname):
    """ a shortcut function to save plot to visualization dir 

    Note
    ----

    We simply assume that every repo will have a 'visulizations' 
    dir under the root directory
    """

    axe = plt.gca()
    plt.savefig(f'visualization/{figname}.pdf',
                format='pdf', dpi=300, bbox_inches='tight')


def save_a_plot(fname, save_dir):
    plt.gcf()
    plt.savefig(fname=os.path.join(save_dir, fname),
                dpi=300,
                format='pdf',
                bbox_inches='tight')
    plt.close()


def cd_root_dir():
    # change directory to the path of the root directory of the project

    ref_path = os.path.abspath('')
    ref_path = pathlib.Path(ref_path).resolve().parents[1]
    os.chdir(ref_path)
    print("current directory:", os.getcwd())


def sin_transformer(period, x):
    return np.sin(x * 2 * np.pi * (1 / period))


def cos_transformer(period, x):
    return np.cos(x * 2 * np.pi * (1 / period))


def pl_residual(gt, preds, low=2, limit=4):
    """ residual plots of deterministic models """

    fig, ax = plt.subplots()
    ax.scatter(preds, gt, color='blue', alpha=0.2)

    # diagonal line
    ax.plot(np.arange(low, limit, 0.01), np.arange(
        low, limit, 0.01), color='gray', ls='--')

    ax.set_xlabel('Predicted revenue in million')
    ax.set_ylabel('Ground truth revenue in million')
    ax.set_title('Ground truth v.s. Predicted revenue')


def pl_residual_horizontal(gt, preds, low=2, limit=4):
    """ residual plots of deterministic models """

    fig, ax = plt.subplots()
    residual = gt - np.squeeze(preds)
    ax.scatter(preds, residual, color='blue', alpha=0.2)
    hl = np.arange(low, limit, 0.01)
    # horizontal line
    ax.plot(hl, np.zeros(shape=(len(hl),)), color='gray', ls='--')
    ax.set_xlabel('Predicted revenue in million')
    ax.set_ylabel('Residual (ground truth - predicted) in million')
    ax.set_title('Residuals v.s. Predicted revenue')



def pl_residual_hist(gt, preds,):
    """ Plot the frequency-dependent residuals

    Parameters
    ----------
    freq_indexx : int, [0,33);
       the index of a frequency axis
    """

    fig, axe = plt.subplots()
    residual = gt - np.squeeze(preds)

    # histogram
    axe = sns.histplot(data=residual, bins=10, binrange=(-1, 1), kde=False, stat='density')
    
    # KDE plot
    # sns.kdeplot(data=data_per_freq, color='crimson', ax=axe)

    mean = np.mean(residual)
    sigma = np.std(residual)

    # fitted Gaussian with MLE paramters
    N = tfp.distributions.Normal(loc=mean, scale=sigma)
    x_axis = np.linspace(-1, 1, 100)
    pdf = N.prob(x_axis)

    axe.plot(x_axis, pdf, color='crimson', linestyle='--')
    # adding such infor in the plot but very hard to control the location dynamically
    # axe.text(0.8, 1., f"mu={mean:.2f}")
    # axe.text(0.8, 1.2, f"std={sigma:.2f}")

    axe.set_xlabel('Residual')
    axe.set_title('Residual distribution (training set)')