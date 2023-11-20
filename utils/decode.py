import numpy as np

def rle_decode(mask_rle, shape=(768, 768)):
    ''' 
    Decode the RLE-encoded mask into an image of a specified shape.

    Args:
    - mask_rle (str): String representing the RLE-encoded mask.
    - shape (tuple, optional): Shape of the image. Default is (768, 768).
    
    Returns:
    - np.ndarray: Decoded image. 
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype = np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def masks_as_image(mask_list):
    """
    Convert a list of RLE-encoded masks into a single image.

    Args:
    - mask_list (list): List of RLE-encoded masks.

    Returns:
    - np.ndarray: Combined image based on the RLE masks.
    """
    all_masks = np.zeros((768, 768), dtype = np.uint8)
    for mask in mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks