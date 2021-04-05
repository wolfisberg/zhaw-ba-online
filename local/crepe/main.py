import os
import tensorflow as tf
import model_crepe
import config
import data


dataset_training = data.get_training_data()
dataset_validation = data.get_validation_data()
dataset_test = data.get_test_data()

model = model_crepe.get_model()
model.summary()

tensorboard_callback = tf.keras.callbacks.TensorBoard(config.LOG_DIR, histogram_freq=1)

if not os.path.exists(config.CHECKPOINT_DIR):
    os.makedirs(config.CHECKPOINT_DIR)
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(config.CHECKPOINT_DIR,'{epoch:02d}-{val_loss:.2f}.hdf5'))

callbacks = [checkpoint, tensorboard_callback]
loss = model.evaluate(dataset_test, steps=1)

history = model.fit(
    dataset_training,
    steps_per_epoch=config.STEPS_PER_EPOCH,
    epochs=config.EPOCHS,
    verbose = 1,
    validation_data = dataset_validation,
    validation_steps=config.VALIDATION_STEPS,
    callbacks = callbacks)
