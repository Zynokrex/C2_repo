import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate2d

def row_fwd_grad(row: list):
    #Np.where, if the elem indx is lower than the length-1, then np.roll(row,-1)-row, if it's like the length-1 then last so val=0
    #np.roll, rolls the row backwards one item (row[1] becomes row[0]) so you can substract directly as we need to do ui+1,j - ui,j
    row_len = len(row)
    return np.where(np.arange(row_len) < row_len - 1, np.roll(row, -1) - row, 0)


def im_fwd_gradient(image: np.ndarray, notacio: str = "profe"):

    grad_i = np.apply_along_axis(row_fwd_grad, 1, image) 
    grad_j = np.apply_along_axis(row_fwd_grad, 1, image.T).T #el mateix que fer-ho per columnes

    if notacio=="profe":
        return grad_i, grad_j
    else: 
        return grad_j, grad_i

def divergence_partial(row: list):
    row_len = len(row)
    #Si es el primer element, ui,j, si esta entre el primer i l'ultim ui,j-ui-1,j (roll 1 posicio endvant es com restar l'anterior)
    #si es l'ultim es posa -ui-1,j
    
    div_parcial = np.where(np.arange(row_len)<1, row,
                            np.where(np.arange(row_len) < row_len - 1, row-np.roll(row,1), -np.roll(row,1)))
    return div_parcial

def im_bwd_divergence(im1: np.ndarray, im2: np.ndarray):

    # CODE TO COMPLETE
    div_i = np.apply_along_axis(divergence_partial, 1, im1)
    div_j = np.apply_along_axis(divergence_partial, 1, im2.T).T
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
    grad_i, grad_j = im_fwd_gradient(u)
    lap_u = im_bwd_divergence(grad_i, grad_j)

    Au = beta * u - lap_u
    return Au

def divergence(vi: np.array, vj: np.array):
    """
    Tiny function to use poisson_editing.divergence(...) from main.py
    """
    return im_bwd_divergence(vi, vj)

def get_translation(source_img: np.ndarray, dst_img: np.ndarray, *part: str):

    # For the eyes mask:
    # The top left pixel of the source mask is located at (x=115, y=101)
    # The top left pixel of the destination mask is located at (x=123, y=125)
    # This gives a translation vector of (dx=8, dy=24)

    # For the mouth mask:
    # The top left pixel of the source mask is located at (x=125, y=140)
    # The top left pixel of the destination mask is located at (x=132, y=173)
    # This gives a translation vector of (dx=7, dy=33)

    corr = correlate2d(dst_img.sum(axis=2), source_img.sum(axis=2), mode='same', boundary="fill", fillvalue=0)
    max_pos = np.unravel_index(np.argmax(corr), corr.shape)
    dy = max_pos[0] - corr.shape[0] // 2
    dx = max_pos[1] - corr.shape[1] // 2
    if part[0] == "eyes":
        print("Difference between hardcoded and computed translation is Y:", np.abs(dy-24), "-- X:", np.abs(dx-8))
        return (24, 8)
    elif part[0] == "mouth":
        print("Difference between hardcoded and computed translation is Y:", np.abs(dy-33), "-- X:", np.abs(dx-7))
        return (33, 7)
    else:
        return (dy, dx)
