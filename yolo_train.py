from math import ceil
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from hand_detector.yolo.darknet import model as yolo_model
from hand_detector.yolo.utils.info import data_info
from hand_detector.yolo.generator import train_generator, valid_generator


def loss_function(y_true, y_pred):

    # binary cross entropy loss
    cross_entropy_loss = tf.keras.losses.binary_crossentropy(y_true[:, :, :, 0:1], y_pred[:, :, :, 0:1])
    cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)
    # mean square loss
    square_diff = tf.math.squared_difference(y_true[:, :, :, 1:5], y_pred[:, :, :, 1:5])
    mask = tf.not_equal(y_true[:, :, :, 1:5], 0)
    mask = tf.cast(mask, tf.float32)
    coordinate_loss = tf.multiply(square_diff, mask)
    coordinate_loss = tf.reduce_sum(coordinate_loss)
    loss = cross_entropy_loss + coordinate_loss
    return loss


# create the model
model = yolo_model()
model.load_weights('weights/yolo.h5')
model.summary()
# compile
adam = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-10)
model.compile(optimizer=adam, loss={"output": loss_function}, metrics={"output": loss_function})

# train
epochs = 100
batch_size = 32
train_set_size = data_info('train')
valid_set_size = data_info('valid')
training_steps_per_epoch = ceil(train_set_size / batch_size)
validation_steps_per_epoch = ceil(valid_set_size / batch_size)
print('training_steps_per_epoch: ', training_steps_per_epoch)
print('validation_steps_per_epoch: ', validation_steps_per_epoch)
train_gen = train_generator(batch_size=batch_size)
valid_gen = valid_generator(batch_size=batch_size)

checkpoints = ModelCheckpoint('weights/yolo_train_best_{epoch:03d}.h5', save_weights_only=True, monitor='val_loss_function',
    mode='min', save_best_only=True)
history = model.fit_generator(train_gen, steps_per_epoch=training_steps_per_epoch, epochs=epochs, verbose=1,
                              validation_data=valid_gen, validation_steps=validation_steps_per_epoch,
                              callbacks=[checkpoints], shuffle=True, max_queue_size=10)

with open('weights/history.txt', 'a+') as f:
    print(history.history, file=f)

print('All Done!')
