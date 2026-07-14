import roifile 
import os 
import holoviews as hv
import numpy as np
import pandas as pd 
from caiman.utils.visualization import get_contours
import caiman as cm
from caiman.base.rois import com as com
import cv2
import matplotlib.pyplot as plt 
import matplotlib as mpl
from itertools import product

    
## Plotting
def ridge_plot(array,times,
               overlay=None,trace_spacing=5,ytick_spacing=1,
               title=None,color='black',text_color = None,
               alpha=0.5,line_width=1,
               ax = None,
               legend=False):
    """
    Plot ridge plot of multiple cells activity.

    Adapted from E. Thomson Caiman Flatiron CCN Workshop 06/2022.

    Parameters
    ----------
    array : ndarray, shape (N, T)
        N cells x T time array of traces (df/f).
    times : ndarray, shape (T,)
        1-D array of time points.
    overlay : ndarray, shape (N, T), optional
        Second trace to plot on top, e.g. a denoised trace.
    trace_spacing : scalar, optional
        Distance between each plot on the y-axis.
    ytick_spacing : int, optional
        Period between y-tick labels (every ytick_spacing traces).
    title : str, optional
        Plot title.
    color : color or list of colors, optional
        Line color: (r, g, b), standard matplotlib color string, or list of
        colors (length must equal number of traces).
    text_color : color, optional
        Color of text and axes.
    alpha : scalar, optional
        Alpha for each trace.
    line_width : scalar, optional
        Width of lines.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes to draw into. Creates new axes if None.
    legend : bool, optional
        Whether to add a legend when plotting two overlaid traces.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object with lines drawn.
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
    """
    Write CNMF spatial components to an ImageJ ROI zip file.

    Parameters
    ----------
    cnmf : CNMF
        CNMF object with estimates.
    idxs : array-like, optional
        Indices of components to write. Writes all components if None.
    savepath : str, optional
        Destination path and filename (must end in .zip).

    Returns
    -------
    savepath : str
        Path to the saved zip file.
    """
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
### load results
data_info_path = '/Users/amonast/Desktop/miniscope/dbnst_data_info.csv'
base_dir_default = '/Volumes/zelikowsky-group2/Michael/Miniscope_Desktop_Backup'

def find_cnmfe_hdf(mouse, session, df, base_dir):
    """
    Find the cnmfe_results.hdf5 file for a given mouse and session.

    Parameters
    ----------
    mouse : str
        Mouse ID, e.g. 'C3M2'.
    session : str
        Session ID, e.g. 'R1'.
    df : DataFrame
        DataFrame with columns ['Mouse', 'Session', 'TSeries_g'].
    base_dir : str
        Root directory containing TSeries folders.

    Returns
    -------
    hdf_path : str
        Full path to cnmfe_results.hdf5.
    """
    row = df[(df['Mouse'] == mouse) & (df['Session'] == session)]
    if row.empty:
        raise ValueError(f'No entry found for Mouse={mouse}, Session={session}')
    tseries = row.iloc[0]['TSeries_g'].strip()
    print(tseries)
    hdf_path = os.path.join(base_dir, tseries, 'suite2p', 'plane0', 'reg_tif', 'cnmfe_results.hdf5')
    if not os.path.exists(hdf_path):
        raise FileNotFoundError(f'HDF5 file not found: {hdf_path}')
    return hdf_path


def load_cnmfe(mouse, session, base_dir=base_dir_default, info_path=data_info_path):
    """
    Load a CNMF-E results object for a given mouse and session.

    Parameters
    ----------
    mouse : str
        Mouse ID, e.g. 'C3M2'.
    session : str
        Session ID, e.g. 'R1'.
    base_dir : str, optional
        Root directory containing TSeries folders.
    info_path : str, optional
        Path to the CSV with columns ['Mouse', 'Session', 'TSeries_g'].

    Returns
    -------
    cnm : CNMF
        Loaded CNMF object with estimates.
    """
    import pandas as pd
    df = pd.read_csv(info_path)
    hdf_path = find_cnmfe_hdf(mouse, session, df, base_dir)
    cnm = cm.source_extraction.cnmf.cnmf.load_CNMF(hdf_path)
    return cnm


#### Plotting functions ##### 
def get_mask(A, idx, dims):
    """
    Convert CNMF spatial contour coordinates into a filled binary mask.

    Parameters
    ----------
    A : scipy.sparse matrix
        cnm.estimates.A spatial components matrix.
    idx : int
        Index of the cell.
    dims : tuple
        Dimensions of the FOV (cnm.estimates.dims).

    Returns
    -------
    mask : ndarray
        Binary mask of the cell's ROI in the FOV.
    """

    coors = cm.utils.visualization.get_contours(A, dims)  # get coordinates of lines around each cell contour
    contour = coors[idx]['coordinates']  # get coordinates for that cell: #verticesx2 array
    c = contour.flatten()  # flatten + remove nans, this step
    c = c[~np.isnan(c)]
    c = c.round().astype(int)  # converting to mask means rounding vertices that are decimals
    c = c.reshape(int(c.shape[0] / 2), 2)  # reshape
    back = np.zeros(dims)  # zeros
    mask = cv2.fillPoly(back, pts=[c], color=(1, 1, 1))

    return mask

def rois_plot_hv(cnm, image=None, idxs=None, min_pct='default', max_pct='default', scale_factor:int=3,
                roi_style='patch', line_width=2, line_color='magenta', display_numbers=False,
                text_size=12, text_color='white', patch_colors=None, patch_alpha=0.2,
                show_bg=True, bgcolor='white'):
    """
    Plot multiple cell ROIs over a background image as patches or contours.

    Parameters
    ----------
    cnm : CNMF
        CNMF object with estimates.
    image : ndarray, optional
        Background image to plot over. If None, no background is rendered
        and bgcolor controls the canvas color.
    idxs : array-like, optional
        Cell indices to plot. Plots all cells if None; plots no cells if -1.
    min_pct : int or 'default', optional
        Minimum percentile of the background image for the colorbar minimum.
    max_pct : int or 'default', optional
        Maximum percentile of the background image for the colorbar maximum.
    scale_factor : int, optional
        Factor by which to scale image dimensions.
    roi_style : {'patch', 'contour'}, optional
        'patch' for translucent patches, 'contour' for contour lines.
    line_width : int, optional
        Width of contour lines (used when roi_style='contour').
    line_color : str, optional
        Color of contour lines (used when roi_style='contour').
    display_numbers : bool, optional
        Whether to display ROI numbers on the plot.
    text_size : int, optional
        Font size of ROI number labels (used when display_numbers=True).
    text_color : str, optional
        Color of ROI number labels (used when display_numbers=True).
    patch_colors : colormap, str, or list of colors, optional
        Colors for patch ROIs (used when roi_style='patch'). If None, all
        patches share the 'spring' colormap on a summed mask. A matplotlib
        colormap (or name string) samples N evenly-spaced colors; a list
        assigns one color per ROI in order. Each ROI is plotted as its own
        transparent overlay.
    patch_alpha : float, optional
        Transparency of patch ROIs (used when roi_style='patch').
    show_bg : bool, optional
        Whether to render the background image. If False, uses a blank canvas
        so bgcolor controls the background color.
    bgcolor : str, optional
        Background color of the plot (default 'white').

    Returns
    -------
    plot : holoviews.element
        HoloViews plot object.
    """
    A = cnm.estimates.A
    dims = cnm.estimates.dims

    if idxs is None:
        idxs = np.arange(0, A.shape[1])

    if idxs.shape[0] == 0:
        masks = np.empty((0, *dims))
        sum_masks = None
    else:
        masks = np.asarray([get_mask(A, i, dims) for i in idxs])
        sum_masks = np.sum(masks, axis=0)
        sum_masks[sum_masks == 0] = np.nan

    if image is None:
        show_bg = False

    clim = im_scale(image, min_pct, max_pct) if show_bg else (0, 1)

    coord = get_contours(cnm.estimates.A, dims=cnm.estimates.dims)
    com, coords = [], []
    for a in idxs:
        com.append(coord[a]['CoM'])
        coords.append(coord[a]['coordinates'][1:-1])
    xs = [p[1] for p in com]
    ys = [p[0] for p in com]
    label_texts = [str(i) for i in idxs]

    w, h = int(cnm.dims[1] * scale_factor), int(cnm.dims[0] * scale_factor)
    if show_bg:
        bg = hv.Image((np.arange(image.shape[1]), np.arange(image.shape[0]), image)).opts(
            invert_yaxis=True, cmap='gray', clim=clim, width=w, height=h)
    else:
        bg = hv.Bounds((0, 0, dims[1], dims[0])).opts(
            line_alpha=0, width=w, height=h, invert_yaxis=True)

    def _patch_overlay():
        if patch_colors is not None and len(masks) > 0:
            cmap_src = mpl.colormaps[patch_colors] if isinstance(patch_colors, str) else patch_colors
            if isinstance(cmap_src, mpl.colors.Colormap):
                cell_colors = [cmap_src(t) for t in np.linspace(0, 1, len(masks))]
            else:
                cell_colors = list(cmap_src)
            overlay = None
            for mask, c in zip(masks, cell_colors):
                r, g, b, _ = mpl.colors.to_rgba(c)
                roi_cmap = mpl.colors.ListedColormap([(r, g, b)])
                mask_nan = mask.astype(float)
                mask_nan[mask_nan == 0] = np.nan
                roi_img = hv.Image((np.arange(dims[1]), np.arange(dims[0]), mask_nan)).opts(
                    cmap=roi_cmap, clim=(0.5, 1.5), alpha=patch_alpha, width=w, height=h, invert_yaxis=True)
                overlay = roi_img if overlay is None else overlay * roi_img
            return overlay
        return hv.Image((np.arange(image.shape[1]), np.arange(image.shape[0]), sum_masks)).opts(
            cmap='spring', alpha=patch_alpha, width=w, height=h, invert_yaxis=True)

    if not display_numbers:
        if roi_style == 'patch':
            plot = bg * _patch_overlay()
        elif roi_style == 'contour':
            contours = hv.Path(coords).opts(color=line_color, invert_yaxis=True, line_width=line_width)
            plot = bg * contours
    else:
        label_overlay = hv.Labels(
            {('x', 'y'): np.stack((np.array(xs), np.array(ys))).T, 'text': label_texts},
            ['x', 'y'], 'text'
        ).opts(text_color=text_color, text_font_size={'value': f'{text_size}px'}, invert_yaxis=True)
        if roi_style == 'patch':
            plot = bg * _patch_overlay() * label_overlay
        elif roi_style == 'contour':
            contours = hv.Path(coords).opts(color=line_color, invert_yaxis=True, line_width=line_width)
            plot = bg * contours * label_overlay

    return plot.opts(bgcolor=bgcolor)


def im_scale(I, min_pct='default', max_pct='default'):
    """
    Set color limits based on image pixel values.

    Parameters
    ----------
    I : ndarray
        Image to be plotted.
    min_pct : {'default', 'min'} or float or int, optional
        Lower color limit. 'default' uses the smallest positive pixel value;
        'min' uses the absolute minimum; a number is treated as a percentile.
    max_pct : 'default' or float or int, optional
        Upper color limit. 'default' uses the image maximum; a number is
        treated as a percentile.

    Returns
    -------
    clim : tuple of (float, float)
        (vmin, vmax) color limits.
    """
    if min_pct == 'default':
        vmin = np.amin(np.unique(I[I > 0]))  # first positive value as min for colormap
    elif min_pct == 'min':
        vmin = np.min(I)
    elif ((type(min_pct) == float) or (type(min_pct) == int)) & (min_pct != 0):
        vmin = np.percentile(I, min_pct)
    elif min_pct == 0:
        vmin = np.min(I)
    if max_pct == 'default':
        vmax = np.max(I)
    elif (type(max_pct) == float) or (type(max_pct) == int):
        vmax = np.percentile(I, max_pct)
    return (vmin, vmax)


def rois_plot(cnm, image=None, idxs=None,
              roi_style='patch', roi_colors='tab20', roi_alpha=0.5,
              line_width=2, min_pct='default', max_pct='default',
              display_numbers=False, text_size=8, text_color='white',
              figsize=None, bgcolor='white', dpi=150, savepath=None,
              return_colors=False):
    """
    Plot cell ROIs over a background image as patches or contours.

    Parameters
    ----------
    cnm : CNMF
        CNMF object with estimates.
    image : ndarray, optional
        Background image. If None, uses a blank background.
    idxs : array-like, optional
        Cell indices to plot. Defaults to all components.
    roi_style : {'patch', 'contour'}, optional
        'patch' for filled translucent patches, 'contour' for outlines.
    roi_colors : colormap, str, or list of colors, optional
        Colors for ROIs. A colormap or name samples N evenly-spaced colors;
        a list assigns one color per ROI.
    roi_alpha : float, optional
        Transparency of ROI patches or contour lines.
    line_width : float, optional
        Width of contour lines (used when roi_style='contour').
    min_pct : int or 'default', optional
        Lower percentile for background image clim.
    max_pct : int or 'default', optional
        Upper percentile for background image clim.
    display_numbers : bool, optional
        Whether to display ROI index numbers at each cell's centre of mass.
    text_size : int, optional
        Font size of ROI number labels.
    text_color : str, optional
        Color of ROI number labels.
    figsize : tuple, optional
        Figure size in inches (width, height). Defaults to image dimensions / 100.
    bgcolor : str, optional
        Background color.
    dpi : int, optional
        Resolution when saving raster formats (PNG, PDF).
    savepath : str, optional
        If provided, saves the figure to this path and closes it.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes with the plot.
    cell_colors : list of RGBA tuples
        Colors assigned to each ROI in idxs order (only if return_colors=True).
    """
    A = cnm.estimates.A
    dims = cnm.estimates.dims

    if idxs is None:
        idxs = np.arange(0, A.shape[1])

    masks = np.asarray([get_mask(A, i, dims) for i in idxs]) if len(idxs) > 0 else np.empty((0, *dims))

    cmap_src = mpl.colormaps[roi_colors] if isinstance(roi_colors, str) else roi_colors
    if isinstance(cmap_src, mpl.colors.Colormap):
        cell_colors = [cmap_src(t) for t in np.linspace(0, 1, max(len(masks), 1))]
    else:
        cell_colors = list(roi_colors)

    if figsize is None:
        figsize = (dims[1] / 100, dims[0] / 100)

    fig, ax = plt.subplots(figsize=figsize, facecolor=bgcolor)
    ax.set_facecolor(bgcolor)

    if image is not None:
        clim = im_scale(image, min_pct, max_pct)
        ax.imshow(image, cmap='gray', vmin=clim[0], vmax=clim[1], origin='upper')

    coord = get_contours(A, dims=dims)

    if roi_style == 'patch':
        for mask, c in zip(masks, cell_colors):
            rgba = np.zeros((*dims, 4))
            r, g, b, _ = mpl.colors.to_rgba(c)
            rgba[mask > 0] = [r, g, b, roi_alpha]
            ax.imshow(rgba, origin='upper')

    elif roi_style == 'contour':
        for idx, c in zip(idxs, cell_colors):
            contour = coord[idx]['coordinates']
            contour = contour[~np.isnan(contour).any(axis=1)]
            ax.plot(contour[:, 0], contour[:, 1], color=c, linewidth=line_width, alpha=roi_alpha)

    if display_numbers:
        for idx, c in zip(idxs, cell_colors):
            cy, cx = coord[idx]['CoM']
            ax.text(cx, cy, str(idx), fontsize=text_size, color=text_color,
                    ha='center', va='center')

    ax.set_xlim(0, dims[1])
    ax.set_ylim(dims[0], 0)
    ax.axis('off')

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches='tight', facecolor=bgcolor)

    if return_colors:
        return ax, cell_colors
    return ax
