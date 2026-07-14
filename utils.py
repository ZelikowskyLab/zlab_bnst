import roifile 
import os 
import numpy as np
from caiman.utils.visualization import get_contours
import matplotlib.pyplot as plt 
import matplotlib as mpl

def ridge_plot(array,times,
               overlay=None,trace_spacing=5,ytick_spacing=1,
               title=None,color='black',text_color = None,
               alpha=0.5,line_width=1,
               ax = None,
               legend=False):
    """
    Plot ridge plot of multiple cells activity Adapted from E. Thomson Caiman Flatiron CCN Workshop 06/2022
    Inputs:
        array(NxT array): # cells N x time array of traces (df/f)
        times (T array): 1-d array
        overlay (NxT array): optional, second trace to plot on top. could be denoised trace.
        trace_spacing (scalar): distance between each plot on y axis
        ytick_spacing (int): period between ytick labels (every ytick_spacing traces)
        color: line color (r,g,b), standard matplotlib color string, or list of colors (len must = # traces plotted)
        alpha (scalar): alpha for each trace, if you want them more see-through when density is high.
        line_width (scalar): width of lines
        ax: axes object if you want to draw in pre-existing axes (None if you want new axes)
        title (string): plot title
        text_color: color of text + axes
        legend (bool): use if plotting two overlaid traces
       
    Outputs:
        ax: axes object with lines drawn
    """
    num_traces = array.shape[0]
    num_yticks = int(num_traces//ytick_spacing)

    # set y position of each trace
    y_position_traces = np.linspace(1, (num_traces)*trace_spacing, num=num_traces)

    # set y tick properties
    y_ticks = np.linspace(1, (num_traces)*trace_spacing, num=num_yticks)
    y_tick_labels = np.arange(1, (num_traces+1)+2*ytick_spacing, ytick_spacing, dtype=np.uint8) # +2*y_tick_spacing just for insurance
    y_tick_labels = y_tick_labels[:num_yticks]

    if ax is None:
        f, ax = plt.subplots()

    #set colors of lines
    for ind, line in enumerate(array):
        floats = np.linspace(0,1,array.shape[0])
        if type(color) is mpl.colors.LinearSegmentedColormap:
            c = color(floats[ind])
        elif (type(color)==np.ndarray) or (type(color)==list):
            c = color[ind]
        else:
            c=color
        dff,=ax.plot(times,line+y_position_traces[ind],color=c,alpha=alpha,linewidth=line_width)
        
        if overlay is not None: #can overlay another trace in a translucent lighter same color
            smooth,=ax.plot(times,overlay[ind,:]+y_position_traces[ind],
                    color=c,alpha=1,linewidth=line_width/2)
   # set text color
    if text_color is not None:
        print('text color')
        mpl.rcParams['text.color'] = text_color
        mpl.rcParams['axes.labelcolor'] = text_color
        mpl.rcParams['xtick.color'] = text_color
        mpl.rcParams['ytick.color'] = text_color
        mpl.rcParams['axes.edgecolor'] = text_color

    # only show left/bottom axis lines
    if legend:
        dff.set_label('Raw dF/F')
        if overlay is not None:
            smooth.set_label('Model Fit')
        #ax.legend(bbox_to_anchor = (2, 0.6))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    return ax

def write_roi_zip(cnmf, idxs=None,savepath='rois.zip'):
    '''
    cnmf: cnmf  object 
    idxs: indices of A to write to file, default None writes all components
    savepath: where to save file and filename (.zip)
    '''
    A = cnmf.estimates.A
    dims = cnmf.estimates.dims
    os.makedirs(os.path.split(savepath)[0],exist_ok=True)
    coords_all = get_contours(A,dims)    
    if idxs is  None:
        idxs = np.arange(0,len(coords_all))
    
    for idx in idxs: 
        coord = coords_all[idx]
        # good_cord = coord[coord]
        roi = roifile.ImagejRoi.frompoints(coord)
        roi.tofile(savepath,name=str(idx))
    
    return savepath