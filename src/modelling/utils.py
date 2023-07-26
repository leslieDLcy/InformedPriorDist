import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt


def save2Vis(figname):
    """ a shortcut function to save plot to visualization dir 
    
    Note
    ----
    
    We simply assume that every repo will have a 'visulizations' 
    dir under the root directory
    """
    
    axe = plt.gca()
    plt.savefig(f'visualization/{figname}.pdf', format='pdf', dpi=300, bbox_inches='tight')


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