import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# Paths
base_dir = 'datasets/combined_dataset_split'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Parameters
img_size = (224, 224)
batch_size = 32
initial_epochs = 10
fine_tune_epochs = 5
total_epochs = initial_epochs + fine_tune_epochs

# Image Generators
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
val_test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')
val_data = val_test_gen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')
test_data = val_test_gen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False)

# Base Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Top Layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(patience=3, restore_best_weights=True)

# 🔁 Initial Training
print("🔁 Initial training (base model frozen)...")
history = model.fit(train_data, validation_data=val_data, epochs=initial_epochs, callbacks=[early_stop])

# 🔓 Unfreeze base model for fine-tuning
print("\n🔓 Fine-tuning base model...")
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# 🔁 Fine-Tuning Training
history_fine = model.fit(train_data, validation_data=val_data, epochs=fine_tune_epochs, callbacks=[early_stop])

# 💾 Save model
model.save("brain_tumor_model.h5")
print("\n✅ Model saved as brain_tumor_model.h5")

# 🧪 Evaluate on test set
loss, acc = model.evaluate(test_data)
print(f"\n🧪 Test Accuracy: {acc * 100:.2f}%")
