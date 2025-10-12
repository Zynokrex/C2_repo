import numpy as np

def row_fwd_grad(row: list):
    #Np.where, if the elem indx is lower than the length-1, then np.roll(row,-1)-row, if it's like the length-1 then last so val=0
    #np.roll, rolls the row backwards one item (row[1] becomes row[0]) so you can substract directly as we need to do ui+1,j - ui,j
    return np.where(np.arange(len(row)) < len(row) - 1, np.roll(row, -1) - row, 0)


def im_fwd_gradient(image: np.ndarray):

    grad_i = np.apply_along_axis(row_fwd_grad, 1, image) 
    grad_j = np.apply_along_axis(row_fwd_grad, 1, image.T).T #el mateix que fer-ho per columnes

    return grad_i, grad_j

def im_bwd_divergence(im1: np.ndarray, im2: np.ndarray):

    # CODE TO COMPLETE
    div_i = 0
    div_j = 0
    return div_i + div_j

def composite_gradients(u1: np.array, u2: np.array, mask: np.array):
    """
    Creates a vector field v by combining the forward gradient of u1 and u2.
    For pixels where the mask is 1, the composite gradient v must coincide
    with the gradient of u1. When mask is 0, the composite gradient v must coincide
    with the gradient of u2.

    :return vi: composition of i components of gradients (vertical component)
    :return vj: composition of j components of gradients (horizontal component)
    """

    # CODE TO COMPLETE
    vi = 0
    vj = 0
    return vi, vj

def poisson_linear_operator(u: np.array, beta: np.array):
    """
    Implements the action of the matrix A in the quadratic energy associated
    to the Poisson editing problem.
    """
    Au = 0
    # CODE TO COMPLETE
    return Au

def get_translation(original_img: np.ndarray, translated_img: np.ndarray, *part: str):

    # For the eyes mask:
    # The top left pixel of the source mask is located at (x=115, y=101)
    # The top left pixel of the destination mask is located at (x=123, y=125)
    # This gives a translation vector of (dx=8, dy=24)

    # For the mouth mask:
    # The top left pixel of the source mask is located at (x=125, y=140)
    # The top left pixel of the destination mask is located at (x=132, y=173)
    # This gives a translation vector of (dx=7, dy=33)

    # The following shifts are hard coded:
    if part[0] == "eyes":
        return (24, 8)
    elif part[0] == "mouth":
        return (33, 7)
    else:
        return (0, 0)

    # Here on could determine the shift vector programmatically,
    # given an original image/mask and its translated version.
    # Idea: using maximal cross-correlation (e.g., scipy.signal.correlate2d), or similar.