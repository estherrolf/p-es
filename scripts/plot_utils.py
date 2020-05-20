import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def reset():
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def setup_always():
    matplotlib.rcParams['pdf.fonttype'] = 42
    
   # plt.rc('font',**{'name':['helvetica']})
    ## for Palatino and other serif fonts use:
    #rc('font',**{'family':'serif','serif':['Palatino']})
    plt.rc('text', usetex=True)
    
def setup_barplot():
    #sns.set_context('paper',font_scale=4)
    #sns.set_style('white')
    matplotlib.rcParams.update({'font.size': 40, 
                                'legend.fontsize':30})
    plt.rc("axes.spines", top=False, right=False)
    
    colors = ['steelblue', 
              #'dodgerblue',
              'orange',
              'darkred']
    patterns = [None, '/','-']
    
    err_line_width=4
    error_kw = {'elinewidth':err_line_width}
    return colors, patterns, error_kw
    
    
def setup_map():
    #sns.set_context('paper',font_scale=4)
    #sns.set_style('white')
    matplotlib.rcParams.update({'font.size': 40, 
                                'legend.fontsize':30})
    plt.rc("axes.spines", top=False, right=False,bottom=False, left=False)
    
    cmap = 'inferno'
    return cmap


def setup_sim_plots():
    setup_always()
    
    sns.set_context('paper',font_scale=4)
    sns.set_style('white')
    matplotlib.rcParams.update({'font.size': 40, 
                                'legend.fontsize':30})

    plt.rc("axes.spines", top=False, right=False)
    #matplotlib.rcParams['image.cmap'] = 'viridis'
    matplotlib.rcParams['image.cmap'] = 'cool'
    
    
def setup_plot_modes_matrix_viz():
    plt.rc("axes.spines", top=False, right=False, bottom=False, left=False)
    #matplotlib.rcParams['image.cmap'] = 'viridis'
    matplotlib.rcParams['image.cmap'] = 'inferno'
    
#     plt.rc("axes.tick_params", False) 

    
