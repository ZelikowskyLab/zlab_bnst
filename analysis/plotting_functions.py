import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import holoviews as hv 
import cv2
import caiman as cm 

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

def rois_plot(footprints,idxs=None,image=None, min_pct='default', max_pct='default', scale_factor: int = 2,
                  cmap_roi='hsv', cmap_image='gray'):
    ''' (Notebook) Plot multiple cells rois over a bacgkround image as patches
    :param footprints: 3d array shape [Ncells,height,width]
    :param image: background image to plot over, optional, if None will just plot footprint
    :param idxs: cell index to plot, *if -1 no cell is plotted*
    :param min_pct: minimum percentile of background image to set as colorbar minimum
    :param max_pct: maximum percentile of background image to set as colorbar maximum
    :param scale_factor: scale dimensions of images
    :param: cmap_roi: colormap for roi
    :param cmap_image: image color map
    :return: Holoviews image overlay object
    '''
    dims = image.shape if image is not None else (footprints.shape[1], footprints.shape[2])
    if idxs is None:  # allows you to plot only some cells
        idxs = np.arange(0, footprints.shape[0])
    sum_masks = np.sum(footprints[idxs, :, :], axis=0)
    sum_masks[sum_masks == 0] = np.nan

    if image is not None:
        clim = im_scale(image, min_pct, max_pct)

        plot = hv.Image(image).opts(cmap=cmap_image, clim=clim, width=int(dims[1] * scale_factor),
                                    height=int(dims[0] * scale_factor),xaxis=None,yaxis=None) * hv.Image(sum_masks).opts(cmap='hsv',
                                                                                                    alpha=.5)
    else:
        plot = hv.Image(sum_masks).opts(cmap=cmap_roi, width=int(dims[1] * scale_factor),
                                        height=int(dims[0] * scale_factor), alpha=.5,xaxis=None,yaxis=None)
    return plot
    
def rois_plot_2(cnm, image=None, idxs=None,
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
    coord = get_contours(A, dims=dims) # this is a caiman function from caiman.utils.visualization.get_contours

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