import argparse

import preprocess
import poisson_editing

import cv2
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

def main():
    parser = argparse.ArgumentParser(description="Poisson blending: choose Lena or Monalisa.")
    parser.add_argument('--image', choices=['lena', 'monalisa'], default='lena',
                        help='Choose which image to blend: lena or monalisa')
    args = parser.parse_args()

    if args.image == 'lena':
        dst, mask, translated_image = preprocess.get_lena(display_images=True)
    elif args.image == 'monalisa':
        dst, mask, translated_image = preprocess.get_monalisa(display_images=True)
    else:
        raise ValueError("Invalid image choice. Use 'lena' or 'monalisa'.")

    u_comb = np.zeros_like(dst) # combined image

    for channel in range(3):
        m = mask[:, :, channel]
        f = dst[:, :, channel]
        u1 = translated_image[:, :, channel]

        beta_0 = 1   # TRY CHANGING
        beta = beta_0 * (1 - m)

        vi, vj = poisson_editing.composite_gradients(u1, f, m)
        b = beta * f - poisson_editing.divergence(vi, vj)
        
        ni, nj = f.shape
        def matvec(x):
            u_img = x.reshape(ni, nj)
            Au = poisson_editing.poisson_linear_operator(u_img, beta)
            return Au.ravel()
        
        A = LinearOperator((ni*nj, ni*nj), matvec=matvec, dtype=np.float64)

        x0 = f.ravel()
        x, info = cg(A, b.ravel(), x0=x0, atol=1e-6, maxiter=800)

        u_comb[:, :, channel] = x.reshape(ni, nj)

    u_final = np.clip(u_comb, 0, 255).astype(np.uint8)

    cv2.imshow('Final result of Poisson blending', u_final)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
