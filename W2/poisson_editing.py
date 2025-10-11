import numpy as np

def im_fwd_gradient(image: np.ndarray):

    # CODE TO COMPLETE
    grad_i = 0
    grad_j = 0
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

    # Get forward gradients of u1 and u2
    g1_i, g1_j = im_fwd_gradient(u1)
    g2_i, g2_j = im_fwd_gradient(u2)

    # Composite gradients using the mask
    vi = mask * g1_i + (1 - mask) * g2_i
    vj = mask * g1_j + (1 - mask) * g2_j

    return vi, vj

def composite_gradients_mixed(u1: np.array, u2: np.array, mask: np.array):
    """
    This function modifies the composite_gradients function to implement
    the "mixed gradients" variant of Poisson editing.

    Creates a vector field v by combining the forward gradient of u1 and u2.
    For pixels where the mask is 1, the composite gradient v must coincide
    with the gradient of u1 if its magnitude is larger than that of u2,
    and vice versa. When mask is 0, the composite gradient v must coincide
    with the gradient of u2.

    :return vi: composition of i components of gradients (vertical component)
    :return vj: composition of j components of gradients (horizontal component)
    """

    # Get forward gradients of u1 and u2
    g1_i, g1_j = im_fwd_gradient(u1)
    g2_i, g2_j = im_fwd_gradient(u2)

    # Select the gradient with the largest magnitude
    vi_in = np.where(np.abs(g2_i) > np.abs(g1_i), g2_i, g1_i)
    vj_in = np.where(np.abs(g2_j) > np.abs(g1_j), g2_j, g1_j)

    # Composite gradients using the mask. Use maximum gradient where mask is 1
    vi = mask * vi_in + (1 - mask) * g2_i
    vj = mask * vj_in + (1 - mask) * g2_j

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