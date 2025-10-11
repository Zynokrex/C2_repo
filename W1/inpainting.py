from dataclasses import dataclass
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import numpy as np

@dataclass
class Parameters:
    hi: float
    hj: float

def laplace_equation(f, mask, param):

    ni = f.shape[0]
    nj = f.shape[1]

    # Add ghost boundaries on the image (for the boundary conditions)
    f_ext = np.zeros((ni + 2, nj + 2), dtype=float)
    ni_ext = f_ext.shape[0]
    nj_ext = f_ext.shape[1]
    f_ext[1: (ni_ext - 1), 1: (nj_ext - 1)] = f

    # Add ghost boundaries on the mask
    mask_ext = np.zeros((ni + 2, nj + 2), dtype=float)
    ndi_ext = mask_ext.shape[0]
    ndj_ext = mask_ext.shape[1]
    mask_ext[1 : ndi_ext - 1, 1 : ndj_ext - 1] = mask

    # Store memory for the A matrix and the b vector
    nPixels = (ni+2)*(nj+2) # Number of pixels

    # We will create A sparse, this is the number of nonzero positions
    # idx_Ai: Vector for the nonZero i index of matrix A
    # idx_Aj: Vector for the nonZero j index of matrix A
    # a_ij: Vector for the value at position ij of matrix A

    b = np.zeros(nPixels, dtype=float)

    # Vector counter
    idx_Ai=[]
    idx_Aj=[]
    a_ij=[]

    # North side boundary conditions
    i = 1
    for j in range(1, nj_ext + 1): 
        p = (j-1)*(ni+2)+i
        
        idx_Ai.append(p)
        idx_Aj.append(p)
        a_ij.append(1)

        idx_Ai.append(p)
        idx_Aj.append(p + 1)
        a_ij.append(-1)

    # South side boundary conditions
    i = ni_ext
    for j in range(1, nj_ext + 1):
        p = (j-1)*(ni+2)+i

        idx_Ai.append(p)
        idx_Aj.append(p)
        a_ij.append(1)

        idx_Ai.append(p)
        idx_Aj.append(p - 1)
        a_ij.append(-1)

    # West side boundary conditions
    j = 1
    for i in range(1, ni_ext + 1):
        p = (j-1)*(ni+2)+i

        idx_Ai.append(p)
        idx_Aj.append(p)
        a_ij.append(1)

        idx_Ai.append(p)
        idx_Aj.append(p + (ni + 2))
        a_ij.append(-1)

    # East side boundary conditions
    j = nj_ext
    for i in range(1, ni_ext + 1):
        p = (j-1)*(ni+2)+i

        idx_Ai.append(p)
        idx_Aj.append(p)
        a_ij.append(1)

        idx_Ai.append(p)
        idx_Aj.append(p - (ni + 2))
        a_ij.append(-1)

    alpha_i = 1.0 / (param.hi**2)
    alpha_j = 1.0 / (param.hj**2)

    # Looping over the pixels
    for j in range(2, nj_ext):
        for i in range(2, ni_ext):

            p = (j-1)*(ni+2)+i

            if mask_ext[i, j] == 1: # we have to in-paint this pixel

                # We need to insert 5 non-zero values into A:
                # u_i,j, u_i+1,j, u_i-1,j, u_i,j+1, u_i,j-1

                # u_i,j
                idx_Ai.append(p)
                idx_Aj.append(p)
                a_ij.append(-(2*alpha_i + 2*alpha_j))

                # u_i+1,j
                idx_Ai.append(p)
                idx_Aj.append(p + 1)
                a_ij.append(alpha_i)

                # u_i-1,j
                idx_Ai.append(p)
                idx_Aj.append(p - 1)
                a_ij.append(alpha_i)

                # u_i,j+1, we move forward a col of the matrix
                idx_Ai.append(p)
                idx_Aj.append(p + (ni + 2))
                a_ij.append(alpha_j)

                # u_i,j-1, we move back a col of the matrix
                idx_Ai.append(p)
                idx_Aj.append(p - (ni + 2))
                a_ij.append(alpha_j)

            else: # we do not have to in-paint this pixel -> we impose u = f

                idx_Ai.append(p)
                idx_Aj.append(p)
                a_ij.append(1)
                b[p - 1] = f_ext[i, j]

    idx_Ai_c = [i - 1 for i in idx_Ai]
    idx_Aj_c = [i - 1 for i in idx_Aj]

    A = sparse(idx_Ai_c, idx_Aj_c, a_ij, nPixels, nPixels)
    x = spsolve(A, b)

    u_ext = np.reshape(x,(ni+2, nj+2), order='F')
    u_ext_i = u_ext.shape[0]
    u_ext_j = u_ext.shape[1]

    u = np.full((ni, nj), u_ext[1:u_ext_i-1, 1:u_ext_j-1], order='F')
    return u

def sparse(i, j, v, m, n):
    """
    Create and compress a matrix that have many zeros
    Parameters:
        i: 1-D array representing the index 1 values
            Size n1
        j: 1-D array representing the index 2 values
            Size n1
        v: 1-D array representing the values
            Size n1
        m: integer representing x size of the matrix >= n1
        n: integer representing y size of the matrix >= n1
    Returns:
        s: 2-D array
            Matrix full of zeros excepting values v at indexes i, j
    """
    return csr_matrix((v, (i, j)), shape=(m, n))