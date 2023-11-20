import os
import cv2
import numpy as np
from utils import decode
from keras.preprocessing.image import ImageDataGenerator
from config import IMG_DATA_GEN_ARGS, IMG_SCALING, BATCH_SIZE, TRAIN_DIR

image_gen = ImageDataGenerator(**IMG_DATA_GEN_ARGS)
label_gen = ImageDataGenerator(**IMG_DATA_GEN_ARGS)

def create_image_gen(input_df, batch_size=BATCH_SIZE):
    '''
    Generates batches of images and their corresponding masks from the input DataFrame.\n
    
    Args:
    - input_df (DataFrame): DataFrame containing image information and encoded masks.
    - batch_size (int, optional): Batch size for the generated images and masks. Default is BATCH_SIZE.
    
    Yields:
    - tuple: A tuple containing batches of images and their respective masks.
    '''
    all_batches = list(input_df.groupby('ImageId'))

    output_rgb_img = []
    output_masks = []

    while True:
        np.random.shuffle(all_batches)

        for curr_img_id, curr_masks in all_batches:
            rgb_path = os.path.join(TRAIN_DIR, curr_img_id)
            curr_img = cv2.imread(rgb_path)
            curr_mask = np.expand_dims(decode.masks_as_image(curr_masks['EncodedPixels'].values), -1)

            if IMG_SCALING is not None:
                curr_img = curr_img[::IMG_SCALING[0], ::IMG_SCALING[1]]
                curr_mask = curr_mask[::IMG_SCALING[0], ::IMG_SCALING[1]]

            output_rgb_img += [curr_img]
            output_masks += [curr_mask]

            if len(output_rgb_img) >= batch_size:
                yield np.stack(output_rgb_img, 0) / 255.0, np.stack(output_masks, 0).astype(np.float32)
                output_rgb_img, output_masks = [], []

def create_aug_gen(input_gen, seed=None):
    '''
    Creates a generator for augmented data based on the input generator.

    Args:
    - input_gen (generator): The input data generator providing input data and labels.
    - seed (int, optional): Seed value for random number generation. 
      If None, a random seed is chosen from the range between 0 and 999. Default is None.
    
    Yields:
    - tuple: A tuple containing augmented input data and labels.
    '''
    np.random.seed(seed if seed is not None else np.random.choice(range(1000)))

    for input_x, input_y in input_gen:
        seed = np.random.choice(range(1000))
        
        x_gen = image_gen.flow(255 * input_x, batch_size=input_x.shape[0],
                               seed=seed, shuffle=True)
        y_gen = label_gen.flow(input_y, batch_size=input_x.shape[0],
                               seed=seed, shuffle=True)
        
        yield next(x_gen) / 255.0, next(y_gen)