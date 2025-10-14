import argparse
import os

import preprocess
import poisson_editing

import cv2
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

def main():
    parser = argparse.ArgumentParser(description="Poisson blending: choose Lena or Monalisa.")
    parser.add_argument('--image', choices=['lena', 'monalisa', 'fire', 'book', 'writing', 'euro'], default='lena',
                        help='Choose which image to blend: lena or monalisa')
    parser.add_argument('--display', action='store_true', default=False,
                        help='Display images during preprocessing (default: do not display)')
    parser.add_argument('--gradients', choices=['default', 'mixed'], default='default',
                        help='Gradient composition method')
    parser.add_argument('--savefinal', action='store_true', default=False,
                    help='Save final blended image to images/outputs/')
    args = parser.parse_args()

    if args.image == 'lena':
        dst, mask, translated_image = preprocess.get_lena(display_images=args.display)
    elif args.image == 'monalisa':
        dst, mask, translated_image = preprocess.get_monalisa(display_images=args.display)
    elif args.image == 'fire':
        dst, mask, translated_image = preprocess.get_fire(display_images=args.display)
    elif args.image == 'book':
        dst, mask, translated_image = preprocess.get_book(display_images=args.display)
    elif args.image == 'writing':
        dst, mask, translated_image = preprocess.get_writing(display_images=args.display)
    elif args.image == 'euro':
        dst, mask, translated_image = preprocess.get_euro(display_images=args.display)
    else:
        raise ValueError("Invalid image choice.")

    u_comb = np.zeros_like(dst) # combined image

    for channel in range(3):
        m = mask[:, :, channel]
        f = dst[:, :, channel]
        u1 = translated_image[:, :, channel]

        beta_0 = 1   # TRY CHANGING
        beta = beta_0 * (1 - m)

        if args.gradients == 'default':
            vi, vj = poisson_editing.composite_gradients(u1, f, m)
        elif args.gradients == 'mixed':
            vi, vj = poisson_editing.composite_gradients_mixed(u1, f, m)
        else:
            raise ValueError("Invalid gradient method")

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

    if args.savefinal:
        output_dir = "images/outputs"
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{args.image}_result_{args.gradients}.png")
        cv2.imwrite(filename, u_final)
        print(f"Saved final image to {filename}")

if __name__ == "__main__":
    main()
