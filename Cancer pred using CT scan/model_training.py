import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers.legacy import Adam


try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if not physical_devices:
        print("No GPU available, using CPU instead.")
        tf.config.set_visible_devices([], 'GPU')
except (RuntimeError, ValueError, UnicodeDecodeError) as e:
    print(f"Error checking for GPU: {e}")

# Define model architecture
base_model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])


optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Set up data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,
    validation_split=0.2,
    fill_mode='nearest'
)

# Create train and validation generators
train_generator = train_datagen.flow_from_directory(
    '/Users/ruthvekkannan/Desktop/python/CT scan/dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    '/Users/ruthvekkannan/Desktop/python/CT scan/dataset/valid',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Learning rate scheduler
def lr_schedule(epoch, lr):
    if epoch > 75:
        lr *= 0.1
    elif epoch > 50:
        lr *= 0.5
    return lr


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-7, verbose=1)


try:
    model_history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=100,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[LearningRateScheduler(lr_schedule), reduce_lr]
    )
except Exception as e:
    print(f"Error during training: {e}")


model_path = '/Users/ruthvekkannan/Desktop/python/CT scan/lung_cancer_model.h5'
model.save(model_path)