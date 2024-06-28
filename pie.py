import numpy as np
import skimage as ski
import scipy
from scipy.sparse.linalg import spsolve, LaplacianNd
from scipy.signal import correlate2d
from numpy.typing import ArrayLike
from typing import Tuple, Optional


hwc2chw = lambda im: im.transpose((2, 0, 1))
chw2hwc = lambda im: im.transpose((1, 2, 0))


def crop_target(
    tgt: np.ndarray, mask: np.ndarray,
    *, tgt_offset: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Parameters:
    tgt: (nchannels, tnrows, tncols)
    mask: (mnrows, mncols)

    Returns:
    tgt_cropped: (nchannels, mnrows, mncols)
    """

    if tgt_offset[0] < 0 or tgt_offset[1] < 0:
        raise ValueError("offsets must be >= 0")
    
    
    if tgt_offset[0] + mask.shape[-2] > tgt.shape[-2] or \
        tgt_offset[1] + mask.shape[-1] > tgt.shape[-1]:
        raise ValueError("selection exceeds `tgt` boundaries")
    
    return tgt[
        :,
        tgt_offset[0] : tgt_offset[0]+mask.shape[-2],
        tgt_offset[1] : tgt_offset[1]+mask.shape[-1]
    ]


def poisson_compositing_wrapper(
    src: np.ndarray, tgt: np.ndarray, mask: np.ndarray, 
    *, tgt_offset: Optional[Tuple[int, int]] = None, mixing: bool = True
) -> np.ndarray:
    """
    Parameters:
    src: shape (mnrows, mncols, nchannels)
    tgt: shape (tnrows, tncols, nchannels)
    mask: shape (mnrows, mncols)

    Returns:
    composite: shape (mnrows, mncols, nchannels)
    """

    assert src.ndim == tgt.ndim == 3
    assert mask.ndim == 2

    if tgt_offset is None:
        tgt_offset = (0, 0)

    src = hwc2chw(src)
    tgt = hwc2chw(tgt)

    tgt_crop = crop_target(tgt, mask, tgt_offset=tgt_offset)

    composite_crop = poisson(src, tgt_crop, mask, mixing=mixing)

    composite = tgt.copy()
    composite[
        :,
        tgt_offset[0] : tgt_offset[0]+mask.shape[-2],
        tgt_offset[1] : tgt_offset[1]+mask.shape[-1]
    ] = composite_crop
    
    return chw2hwc(composite)


def laplacian_matrix(n: int, m: int) -> scipy.sparse.csr_array:
    """
    Returns:
    L: laplacian matrix with shape (n*m, n*m), csr_array

    See Wikipedia article for notation:
    https://en.wikipedia.org/wiki/Discrete_Poisson_equation#On_a_two-dimensional_rectangular_grid
    """

    D = scipy.sparse.lil_array((m, m))
    D.setdiag(-1, -1)
    D.setdiag(4)
    D.setdiag(-1, 1)

    A = scipy.sparse.block_diag([D] * n, format="csr")
    A.setdiag(-1, m)
    A.setdiag(-1, -m)

    return A


def diff(gray: np.ndarray, ker: ArrayLike, *, axis: int) -> np.ndarray:
    """
    Parameters:
    gray: single channel image, shape (nrows, ncols)
    ker: finite difference kernel, shape (3,)

    
    Returns:
    2D cross-correlation of `gray` with (a 2D version of) `ker`
    """

    if axis not in range(-2, 2):
        raise ValueError("Wrong `axis` value.")
    
    ker = np.asarray(ker)
    if ker.shape != (3,):
        raise ValueError("`ker` must have shape (3,).")

    kernel = np.zeros((3, 3))
    kernel[1, :] = ker

    if axis == -2 or axis == 0:
        kernel = kernel.T

    return correlate2d(gray, kernel, mode="same")


def poisson(src: np.ndarray, tgt: np.ndarray, region: np.ndarray, mixing : bool = True) -> np.ndarray:
    """
    Parameters:
    src: source image with values in [0, 1], shape (nchannels, nrows, ncols)
    tgt: target image with values in [0, 1], shape (nchannels, nrows, ncols)
    region: binary mask of the region to clone, shape (nrows, ncols)
    mixing: use mixing gradients approach 

    Returns:
    out: composite image, shape (nchannels, nrows, ncols)
    """
    assert (
        src.shape == tgt.shape
    ), f"src and tgt shapes must be equal, {src.shape=} != {tgt.shape=}"
    assert (
        src.shape[-2:] == tgt.shape[-2:] == region.shape
    ), f"src, tgt and region last two dims must be equal, {src.shape=}, {tgt.shape=}, {region.shape=}"
    assert region.dtype == np.bool, "`region` must be a boolean mask."

    nchannels, nrows, ncols = tgt.shape

    ninside = len(np.flatnonzero(region))  # n. of positions inside region

    # get laplacian entries only for positions inside region
    # L = laplacian_matrix(nrows, ncols)[region.reshape(nrows*ncols), :] # (ninside, nrows*ncols), csr_array
    L = -LaplacianNd(
        (nrows, ncols), 
        boundary_conditions="dirichlet"
    ).tosparse()[region.reshape(nrows * ncols), :]  # (ninside, nrows*ncols), dia_array

    # filter out coefficients for neighbours outside region
    A = L[:, region.reshape(nrows * ncols)]  # (ninside, ninside)

    b = np.zeros((nchannels, ninside))

    if mixing:
        gradx_src = np.zeros((nchannels, nrows, ncols))  # gradient along the x-axis
        grady_src = np.zeros((nchannels, nrows, ncols))  # gradient along the y-axis
        gradmag_src = np.zeros(
            (nchannels, nrows, ncols)
        )  # gradient magnitude (not normalized)

        gradx_tgt = np.zeros((nchannels, nrows, ncols))
        grady_tgt = np.zeros((nchannels, nrows, ncols))
        gradmag_tgt = np.zeros((nchannels, nrows, ncols))

        for channel in range(nchannels):
            gradx_src[channel] = diff(src[channel], [-1, 1, 0], axis=-1)
            grady_src[channel] = diff(src[channel], [-1, 1, 0], axis=-2)
            gradmag_src[channel] = gradx_src[channel] ** 2 + grady_src[channel] ** 2

            gradx_tgt[channel] = diff(tgt[channel], [-1, 1, 0], axis=-1)
            grady_tgt[channel] = diff(tgt[channel], [-1, 1, 0], axis=-2)
            gradmag_tgt[channel] = gradx_tgt[channel] ** 2 + grady_tgt[channel] ** 2

        guidx = np.where(gradmag_tgt > gradmag_src, gradx_tgt, gradx_src)  # (nchannels, nrows, ncols)
        guidy = np.where(gradmag_tgt > gradmag_src, grady_tgt, grady_src)

        del gradx_src, grady_src, gradmag_src, gradx_tgt, grady_tgt, gradmag_tgt

        for channel in range(nchannels):
            b[channel] = (diff(guidx[channel], [0, 1, -1], axis=-1) + diff(guidy[channel], [0, 1, -1], axis=-2))[region]  # divergence

        del guidx, guidy

    else:
        for channel in range(nchannels):
            b[channel, :] = L.dot(src[channel].reshape(nrows * ncols))

    eroded = ski.morphology.binary_erosion(
        region,
        footprint=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),  # 4-connected neighbours
    )

    # positions (r, c) which have some neigbours outside region
    nbrsout = np.logical_xor(region, eroded)  # (nrows, ncols)

    tgt_zeroed = np.copy(tgt)
    tgt_zeroed[:, region] = 0.0

    tgt_zeroed_filtered = np.zeros_like(tgt_zeroed)
    for channel in range(nchannels):
        tgt_zeroed_filtered[channel] = correlate2d(
            tgt_zeroed[channel],
            np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
            mode="same",
        )

    # get relative indexes
    rel_idxs = np.flatnonzero(nbrsout[region])

    for channel in range(nchannels):
        b[channel, rel_idxs] += tgt_zeroed_filtered[channel, region][rel_idxs]

    out = tgt_zeroed.reshape(nchannels, nrows * ncols)
    for channel in range(nchannels):
        x = spsolve(A, b[channel])
        x[x > 1.0] = 1.0
        x[x < 0.0] = 0.0
        out[channel, region.reshape(nrows * ncols)] = x

    return out.reshape(nchannels, nrows, ncols)
