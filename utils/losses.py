import keras.backend as k
from config import ALPHA, GAMMA

def focal_loss(targets, inputs, alpha=ALPHA, gamma=GAMMA):
    '''
    Computes the Focal Loss between predicted 'inputs' and target 'targets'.

    Focal Loss is designed to address class imbalance in binary classification tasks.
    
    Parameters:
    - targets (tensor): True values.
    - inputs (tensor): Model predictions.
    - alpha (float, optional): Weighting factor (default is ALPHA).
    - gamma (float, optional): Power factor (default is GAMMA).

    Returns:
    - loss: focal loss value.
    '''
    inputs = k.flatten(inputs)
    targets = k.flatten(targets)

    bce = k.binary_crossentropy(targets, inputs)
    bce_exp = k.exp(-bce)
    loss = k.mean(alpha * k.pow((1- bce_exp), gamma) * bce)

    return loss

def dice_score(y_true, y_pred, smooth=1):
    """
    Calculates the dice score to quantify similarity between two images.

    Args:
    - y_true: True values.
    - y_pred: Predicted values.
    - smooth: Parameter to prevent division by zero.

    Returns:
    - dice_score: dice score, measuring image similarity.
    """
    intersection = k.sum(y_true * y_pred, axis=[1,2,3])
    union = k.sum(y_true, axis=[1,2,3]) + k.sum(y_pred, axis=[1,2,3])
    dice_score = k.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice_score