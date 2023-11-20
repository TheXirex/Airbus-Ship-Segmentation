import pandas as pd
from keras import layers
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from config import *
from utils import losses, generators, unet

df = pd.read_csv(DATASET_DIR)

# Counting the number of ships in images by unique id
df['Ships'] = df['EncodedPixels'].apply(lambda row: 1 if isinstance(row, str) else 0)
unique_ids = df.groupby('ImageId').agg({'Ships': 'sum'}).reset_index()
df.drop(['Ships'], axis=1, inplace=True)


# Balancing dataset
balanced_train_df = unique_ids.groupby('Ships').apply(
    lambda x: x.sample(SAMPLES_PER_GROUP) if len(x) > SAMPLES_PER_GROUP else x)

print('Number of samples per group:\n', # (group == number of ships in the image)
      balanced_train_df.groupby(level=0).size())


# Splitting train and validation data
train_ids, valid_ids = train_test_split(balanced_train_df,
                                        test_size=0.2,
                                        stratify=balanced_train_df['Ships'])

train_df = pd.merge(df, train_ids)
valid_df = pd.merge(df, valid_ids)

print(f'Count of training masks: {train_df.shape[0]}.\n',
      f'Count of validation masks: {valid_df.shape[0]}.')

# creating model
model = unet.Unet()


# set up the callbacks for training
checkpoint = ModelCheckpoint(filepath="weights_and_models/best.h5", 
                             monitor='val_dice_score', verbose=1, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_dice_score', factor=0.2, patience=3,
                              verbose=1, mode='max', min_delta=0.0001, cooldown=2, min_lr=1e-6)
early_stop = EarlyStopping(monitor="val_dice_score", mode="max", patience=15)

callbacks_list = [checkpoint, early_stop, reduce_lr]


# compiling & training model
model.compile(optimizer=Adam(1e-3), loss=losses.focal_loss, metrics=[losses.dice_score])

aug_gen = generators.create_aug_gen(generators.create_image_gen(train_df))
valid_x, valid_y = next(generators.create_image_gen(valid_df, VALID_IMG_COUNT))

step_count = min(MAX_TRAIN_STEPS, train_df.shape[0] // BATCH_SIZE)

loss_history = model.fit(aug_gen,
                         epochs=MAX_TRAIN_EPOCHS,
                         steps_per_epoch=step_count,
                         validation_data=(valid_x, valid_y),
                         callbacks=callbacks_list, workers=1)


# save results
loss_df = pd.DataFrame(loss_history.history)
loss_df.to_csv('models_history/history_model.csv', index=False)
model.save('weights_and_models/last.h5')