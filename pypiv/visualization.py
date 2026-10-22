import numpy as np
import matplotlib.pyplot as plt


def velocity_histogram(u: np.ndarray, v: np.ndarray, ax=None, show=False, **histogram2d_kwargs):
    hist, x_edges, y_edges = np.histogram2d(u.ravel(), v.ravel(), **histogram2d_kwargs)

    if ax is None: 
        _, ax = plt.subplots()

    im = ax.pcolormesh(x_edges, y_edges, hist.T)
    ax.set_xlabel(r"$u$ [m s⁻¹]")
    ax.set_ylabel(r"$v$ [m s⁻¹]")

    if show:
        plt.show()
        return (hist, x_edges, y_edges)
    
    else:
        return im, (hist, x_edges, y_edges)

    
def piv_plot(x, y, u, v, flags=None, background=None, ax=None, scale=None, show=False):
    mask = np.zeros_like(u) if flags is None else flags
    if ax is None:
        _, ax = plt.subplots(dpi=100)
    if background is not None:
        ax.pcolormesh(background, cmap="gray", rasterized=True)
    
    ax.quiver(x, y, u * (1 - mask), v * (1 - mask), scale=scale)
    ax.set_aspect('equal')
    
    if show: plt.show()
    return ax


def compare_frames(frame_a, frame_b, ax=None, show=False):

    overlay = np.zeros((*frame_a.shape, 3))
    overlay[..., 0] = frame_a
    overlay[..., 2] = frame_b
    overlay[..., 1] = np.zeros_like(frame_a)

    if ax is None:
        _, ax = plt.subplots(dpi=100)

    ax.imshow(overlay, origin="lower")
    if show:
        plt.show()
    else:
        return ax

    
def validation_check(u, v, u_filtered, v_filtered):
    _, axs = plt.subplots(1, 2)

    axs[0].set_title("Original")
    im1, _ = velocity_histogram(u, v, ax=axs[0], density=True)
    plt.colorbar(im1)

    axs[1].set_title("Filtered")
    im2, _ = velocity_histogram(u_filtered, v_filtered, ax=axs[1], density=True)
    plt.colorbar(im2)

    plt.show()