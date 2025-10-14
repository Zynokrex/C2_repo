import numpy as np
import cv2
import poisson_editing

def shift_source(src, src_masks, translations, ni, nj, nChannels, padding=5):
    translated_image = src.copy()
    for t, src_mask in zip(translations, src_masks):
        dy, dx = t
        for c in range(nChannels):
            mask_channel = src_mask[:, :, c] // 255

            # Find indices (points) where the mask is active
            indices = np.where(mask_channel != 0)
            if indices[0].size == 0:
                continue

            i_min, i_max = indices[0].min(), indices[0].max()
            j_min, j_max = indices[1].min(), indices[1].max()

            top    = max(0, i_min - padding)
            bottom = min(ni, i_max + 1 + padding)   # +1: inclusive -> exclusive
            left   = max(0, j_min - padding)
            right  = min(nj, j_max + 1 + padding)

            # Build full rectangular grid of the crop region
            rect_i = np.arange(top, bottom, dtype=np.int32)
            rect_j = np.arange(left, right, dtype=np.int32)
            grid_i, grid_j = np.meshgrid(rect_i, rect_j, indexing='ij')

            # Shifted positions
            shifted_i = grid_i + dy
            shifted_j = grid_j + dx

            # Only keep indices inside image bounds
            valid = (shifted_i >= 0) & (shifted_i < ni) & (shifted_j >= 0) & (shifted_j < nj)

            # Paste the rectangular crop (including padding/context)
            translated_image[shifted_i[valid], shifted_j[valid], c] = src[grid_i[valid], grid_j[valid], c]
    return translated_image

def get_lena(display_images = False):
    """
    Function to load the images and masks for the eye and mouth swapping
    example with Lena.
    """
    # Load images
    src = cv2.imread('images/lena/girl.png').astype(np.float64)
    dst = cv2.imread('images/lena/lena.png').astype(np.float64)

    # Store shapes and number of channels (src, dst and mask should have same dimensions!)
    ni, nj, nChannels = dst.shape
    if display_images:
        # Display the images
        cv2.imshow('Source image', np.clip(src, 0, 255).astype(np.uint8)); cv2.waitKey(0)
        cv2.imshow('Destination image', np.clip(dst, 0, 255).astype(np.uint8)); cv2.waitKey(0)

    # Load masks for eye swapping
    src_mask_eyes = cv2.imread('images/lena/mask_src_eyes.png', cv2.IMREAD_COLOR)
    dst_mask_eyes = cv2.imread('images/lena/mask_dst_eyes.png', cv2.IMREAD_COLOR)
    if display_images:
        cv2.imshow('Eyes source mask', np.clip(src_mask_eyes, 0, 255).astype(np.uint8)); cv2.waitKey(0)
        cv2.imshow('Eyes destination mask', np.clip(dst_mask_eyes, 0, 255).astype(np.uint8)); cv2.waitKey(0)

    # Load masks for mouth swapping
    src_mask_mouth = cv2.imread('images/lena/mask_src_mouth.png', cv2.IMREAD_COLOR)
    dst_mask_mouth = cv2.imread('images/lena/mask_dst_mouth.png', cv2.IMREAD_COLOR)
    if display_images:
        cv2.imshow('Mouth source mask', np.clip(src_mask_mouth, 0, 255).astype(np.uint8)); cv2.waitKey(0)
        cv2.imshow('Mouth destination mask', np.clip(dst_mask_mouth, 0, 255).astype(np.uint8)); cv2.waitKey(0)

    # Get the translation vectors (hard coded)
    t_eyes = poisson_editing.get_translation(src_mask_eyes, dst_mask_eyes, "eyes")
    t_mouth = poisson_editing.get_translation(src_mask_mouth, dst_mask_mouth, "mouth")

    translations = [t_eyes, t_mouth]
    src_masks = [src_mask_eyes, src_mask_mouth]
    translated_image = shift_source(src, src_masks, translations, ni, nj, nChannels)
    if display_images:
        cv2.imshow('Source image after shifting', np.clip(translated_image, 0, 255).astype(np.uint8)); cv2.waitKey(0)
    u_comb = dst.copy()
    mask = np.zeros_like(dst) # combined mask
    # Blend with the original (destination) image
    for dst_mask in [dst_mask_eyes, dst_mask_mouth]:
        for c in range(nChannels):
            mask_channel = dst_mask[:, :, c] // 255  # Convert to binary mask
            mask[:, :, c] = np.where(mask_channel != 0, 1, mask[:, :, c])
            u_comb[:, :, c] = np.where(mask_channel != 0, translated_image[:, :, c], u_comb[:, :, c])

    if display_images:
        cv2.imshow('Combined image just by copying', np.clip(u_comb, 0, 255).astype(np.uint8)); cv2.waitKey(0)
            
    return dst, mask, translated_image


def get_monalisa(display_images = False):
    """
    Function to load the images and masks for the eye swapping
    example with Mona Lisa.
    """
    # Load images
    src = cv2.imread('images/monalisa/ginevra.png').astype(np.float64)
    dst = cv2.imread('images/monalisa/lisa.png').astype(np.float64)

    # Store shapes and number of channels (src, dst and mask should have same dimensions!)
    _, _, nChannels = dst.shape

    # Display the images
    if display_images:
        cv2.imshow('Source image', np.clip(src, 0, 255).astype(np.uint8)); cv2.waitKey(0)
        cv2.imshow('Destination image', np.clip(dst, 0, 255).astype(np.uint8)); cv2.waitKey(0)

    # Load masks for eye swapping
    src_dst_mask = cv2.imread('images/monalisa/mask.png', cv2.IMREAD_COLOR)
    if display_images:
        cv2.imshow('Source and destination mask', np.clip(src_dst_mask, 0, 255).astype(np.uint8)); cv2.waitKey(0)

    # Since translation is (0,0) we do not need to call shift_source, we use src instead of translated_image
    u_comb = dst.copy()
    mask = np.zeros_like(dst) # combined mask
    # Blend with the original (destination) image
    for dst_mask in [src_dst_mask]:
        for c in range(nChannels):
            mask_channel = dst_mask[:, :, c] // 255  # Convert to binary mask
            mask[:, :, c] = np.where(mask_channel != 0, 1, mask[:, :, c])
            u_comb[:, :, c] = np.where(mask_channel != 0, src[:, :, c], u_comb[:, :, c])

    if display_images:
        cv2.imshow('Combined image just by copying', np.clip(u_comb, 0, 255).astype(np.uint8)); cv2.waitKey(0)

    return dst, mask, src


def get_fire(display_images = False):
    """
    Function to load the images and masks for the fire example.
    """
    # Load images
    src = cv2.imread('images/fire/fire.png').astype(np.float64)
    dst = cv2.imread('images/fire/wood.png').astype(np.float64)

    # Store shapes and number of channels (src, dst and mask should have same dimensions!)
    _, _, nChannels = dst.shape

    # Display the images
    if display_images:
        cv2.imshow('Source image', np.clip(src, 0, 255).astype(np.uint8)); cv2.waitKey(0)
        cv2.imshow('Destination image', np.clip(dst, 0, 255).astype(np.uint8)); cv2.waitKey(0)

    # Load masks for eye swapping
    src_dst_mask = cv2.imread('images/fire/mask.png', cv2.IMREAD_COLOR)
    if display_images:
        cv2.imshow('Source and destination mask', np.clip(src_dst_mask, 0, 255).astype(np.uint8)); cv2.waitKey(0)

    # Since translation is (0,0) we do not need to call shift_source, we use src instead of translated_image
    u_comb = dst.copy()
    mask = np.zeros_like(dst) # combined mask
    # Blend with the original (destination) image
    for dst_mask in [src_dst_mask]:
        for c in range(nChannels):
            mask_channel = dst_mask[:, :, c] // 255  # Convert to binary mask
            mask[:, :, c] = np.where(mask_channel != 0, 1, mask[:, :, c])
            u_comb[:, :, c] = np.where(mask_channel != 0, src[:, :, c], u_comb[:, :, c])

    if display_images:
        cv2.imshow('Combined image just by copying', np.clip(u_comb, 0, 255).astype(np.uint8)); cv2.waitKey(0)

    return dst, mask, src


def get_book(display_images = False):
    """
    Function to load the images and masks for the book example.
    """
    # Load images
    src = cv2.imread('images/book/hand.png').astype(np.float64)
    dst = cv2.imread('images/book/book.png').astype(np.float64)

    # Store shapes and number of channels (src, dst and mask should have same dimensions!)
    _, _, nChannels = dst.shape

    # Display the images
    if display_images:
        cv2.imshow('Source image', np.clip(src, 0, 255).astype(np.uint8)); cv2.waitKey(0)
        cv2.imshow('Destination image', np.clip(dst, 0, 255).astype(np.uint8)); cv2.waitKey(0)

    # Load masks for eye swapping
    src_dst_mask = cv2.imread('images/book/mask.png', cv2.IMREAD_COLOR)
    if display_images:
        cv2.imshow('Source and destination mask', np.clip(src_dst_mask, 0, 255).astype(np.uint8)); cv2.waitKey(0)

    # Since translation is (0,0) we do not need to call shift_source, we use src instead of translated_image
    u_comb = dst.copy()
    mask = np.zeros_like(dst) # combined mask
    # Blend with the original (destination) image
    for dst_mask in [src_dst_mask]:
        for c in range(nChannels):
            mask_channel = dst_mask[:, :, c] // 255  # Convert to binary mask
            mask[:, :, c] = np.where(mask_channel != 0, 1, mask[:, :, c])
            u_comb[:, :, c] = np.where(mask_channel != 0, src[:, :, c], u_comb[:, :, c])

    if display_images:
        cv2.imshow('Combined image just by copying', np.clip(u_comb, 0, 255).astype(np.uint8)); cv2.waitKey(0)

    return dst, mask, src


def get_writing(display_images = False):
    """
    Function to load the images and masks for the book example.
    """
    # Load images
    src = cv2.imread('images/writing/writing.png').astype(np.float64)
    dst = cv2.imread('images/writing/book.png').astype(np.float64)

    # Store shapes and number of channels (src, dst and mask should have same dimensions!)
    _, _, nChannels = dst.shape

    # Display the images
    if display_images:
        cv2.imshow('Source image', np.clip(src, 0, 255).astype(np.uint8)); cv2.waitKey(0)
        cv2.imshow('Destination image', np.clip(dst, 0, 255).astype(np.uint8)); cv2.waitKey(0)

    # Load masks for eye swapping
    src_dst_mask = cv2.imread('images/writing/mask.png', cv2.IMREAD_COLOR)
    if display_images:
        cv2.imshow('Source and destination mask', np.clip(src_dst_mask, 0, 255).astype(np.uint8)); cv2.waitKey(0)

    # Since translation is (0,0) we do not need to call shift_source, we use src instead of translated_image
    u_comb = dst.copy()
    mask = np.zeros_like(dst) # combined mask
    # Blend with the original (destination) image
    for dst_mask in [src_dst_mask]:
        for c in range(nChannels):
            mask_channel = dst_mask[:, :, c] // 255  # Convert to binary mask
            mask[:, :, c] = np.where(mask_channel != 0, 1, mask[:, :, c])
            u_comb[:, :, c] = np.where(mask_channel != 0, src[:, :, c], u_comb[:, :, c])

    if display_images:
        cv2.imshow('Combined image just by copying', np.clip(u_comb, 0, 255).astype(np.uint8)); cv2.waitKey(0)

    return dst, mask, src


def get_euro(display_images = False):
    """
    Function to load the images and masks for the book example.
    """
    # Load images
    src = cv2.imread('images/euro/spongebob.png').astype(np.float64)
    dst = cv2.imread('images/euro/euro.png').astype(np.float64)

    # Store shapes and number of channels (src, dst and mask should have same dimensions!)
    _, _, nChannels = dst.shape

    # Display the images
    if display_images:
        cv2.imshow('Source image', np.clip(src, 0, 255).astype(np.uint8)); cv2.waitKey(0)
        cv2.imshow('Destination image', np.clip(dst, 0, 255).astype(np.uint8)); cv2.waitKey(0)

    # Load masks for eye swapping
    src_dst_mask = cv2.imread('images/euro/mask.png', cv2.IMREAD_COLOR)
    if display_images:
        cv2.imshow('Source and destination mask', np.clip(src_dst_mask, 0, 255).astype(np.uint8)); cv2.waitKey(0)

    # Since translation is (0,0) we do not need to call shift_source, we use src instead of translated_image
    u_comb = dst.copy()
    mask = np.zeros_like(dst) # combined mask
    # Blend with the original (destination) image
    for dst_mask in [src_dst_mask]:
        for c in range(nChannels):
            mask_channel = dst_mask[:, :, c] // 255  # Convert to binary mask
            mask[:, :, c] = np.where(mask_channel != 0, 1, mask[:, :, c])
            u_comb[:, :, c] = np.where(mask_channel != 0, src[:, :, c], u_comb[:, :, c])

    if display_images:
        cv2.imshow('Combined image just by copying', np.clip(u_comb, 0, 255).astype(np.uint8)); cv2.waitKey(0)

    return dst, mask, src