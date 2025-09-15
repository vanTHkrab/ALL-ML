import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import cv2

# --- 1. CONFIGURATION AND SETUP ---
# Update these paths to match your local computer's file structure
DATA_DIR = "./data/prepared-2"
VOLUME_FILES_DIR = "data/volumes"
MODEL_SAVE_PATH = "./models/mangosteen_volume_model_all_suggessions_2.h5" # Saves in the current working directory

# Model parameters
IMG_SIZE = 224
BATCH_SIZE = 32
INITIAL_EPOCHS = 30 # Reduced for initial head training
FINE_TUNE_EPOCHS = 70 # Increased for fine-tuning the backbone
LEARNING_RATE_INITIAL = 1e-3
LEARNING_RATE_FINE_TUNE = 1e-5

# --- 2. DATA LOADING AND PREPARATION ---

def load_data_from_folders(data_dir, volume_dir, allowed_prefix_pattern=r'^(i\d{3})', verbose=True):
    """
    โหลดภาพและค่า volume โดยจับคู่ด้วย prefix แบบ i001*, i002*, ...
    - ภาพใด ๆ ที่ขึ้นต้นด้วย (i + เลข 3 หลัก) จะถูก map ไปหา <prefix>.txt ใน volume_dir
    - ถ้ามีหลายภาพต่อ 1 prefix จะนับเป็นหลาย sample โดยใช้ค่า volume เดียวกัน

    Args:
        data_dir (str): โฟลเดอร์ภาพ
        volume_dir (str): โฟลเดอร์ไฟล์ volume (.txt) ที่ตั้งชื่อตาม prefix (เช่น i001.txt)
        allowed_prefix_pattern (str): regex สำหรับดึง prefix (default: r'^(i\\d{3})')
        verbose (bool): พิมพ์สรุป/คำเตือน

    Returns:
        tuple[list[str], list[float]]: (image_paths, volumes) ที่จับคู่สำเร็จแล้ว
    """
    image_paths, volumes = [], []

    if not os.path.exists(data_dir):
        print(f"❌ Image data directory not found: {data_dir}")
        return [], []
    if not os.path.exists(volume_dir):
        print(f"❌ Volume data directory not found: {volume_dir}")
        return [], []

    prefix_re = re.compile(allowed_prefix_pattern, re.IGNORECASE)

    # เก็บสถานะเพื่อตรวจว่า prefix ไหนไม่มีไฟล์ .txt
    seen_prefixes = set()
    missing_volume_for_prefix = set()

    for filename in os.listdir(data_dir):
        lower = filename.lower()
        if not lower.endswith(('.png', '.jpg', '.jpeg')):
            continue

        m = prefix_re.match(os.path.splitext(filename)[0])
        if not m:
            if verbose:
                print(f"⚠️  Skip (no valid prefix): {filename}")
            continue

        prefix = m.group(1).lower()
        seen_prefixes.add(prefix)

        volume_filename = f"{prefix}_mangosteen_grid.txt"
        volume_filepath = os.path.join(volume_dir, volume_filename)

        if not os.path.exists(volume_filepath):
            missing_volume_for_prefix.add(prefix)
            if verbose:
                print(f"⚠️  Missing volume file for prefix '{prefix}': expected {volume_filename}")
            continue

        try:
            with open(volume_filepath, 'r') as f:
                # รองรับไฟล์ที่มีตัวเลขหลายบรรทัด โดยจะเอาบรรทัดแรกที่ parse ได้
                vol = None
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    vol = float(s)
                    break
                if vol is None:
                    raise ValueError("No numeric value found")
        except Exception as e:
            if verbose:
                print(f"⚠️  Bad volume file for {prefix}: {volume_filename} (error: {e})")
            continue

        # ผ่านหมด → บันทึกเป็น sample หนึ่งตัว (ภาพหนึ่งใบต่อหนึ่งแถว)
        image_paths.append(os.path.join(data_dir, filename))
        volumes.append(vol)

    if verbose:
        print("\n— Matching summary —")
        print(f"Prefixes detected: {len(seen_prefixes)}")
        print(f"Samples matched (images with volume): {len(image_paths)}")
        if missing_volume_for_prefix:
            print(f"Prefixes missing .txt: {sorted(missing_volume_for_prefix)[:10]}{' ...' if len(missing_volume_for_prefix) > 10 else ''}")

    return image_paths, volumes

print("Loading data...")
image_paths, volumes = load_data_from_folders(DATA_DIR, VOLUME_FILES_DIR)

if not image_paths:
    print("❌ No matching image and volume data found. Please check your paths and file names.")
    exit()

# Split data into training and validation sets
train_paths, val_paths, train_volumes, val_volumes = train_test_split(
    image_paths, volumes, test_size=0.2, random_state=42
)

print(f"Total samples found: {len(image_paths)}")
print(f"Training samples: {len(train_paths)}")
print(f"Validation samples: {len(val_paths)}")


# --- Data Augmentation function for brightness and contrast ---
def augment_image(image):
    # Convert image to float32 for augmentation
    image = tf.convert_to_tensor(image, dtype=tf.float32)

    # Randomly adjust brightness by a factor between -0.2 and 0.2
    image = tf.image.random_brightness(image, max_delta=0.2)
    
    # Randomly adjust contrast by a factor between 0.8 and 1.2
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # Clip the augmented image to the valid range [0, 255]
    return tf.clip_by_value(image, 0.0, 255.0)


# Data Generators
# We use a custom generator to load images and their corresponding volumes
def data_generator(image_paths, volumes, batch_size, is_training=True):
    num_samples = len(image_paths)
    while True:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]
            
            batch_images = []
            batch_volumes = []
            
            for i in batch_indices:
                img = cv2.imread(image_paths[i])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Ensure correct color format
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) # Resize to model input size
                batch_images.append(img)
                batch_volumes.append(volumes[i])
            
            # Apply augmentation only to the training set
            if is_training:
                batch_images = augment_image(np.array(batch_images, dtype="float32"))
            else:
                batch_images = np.array(batch_images, dtype="float32")
            
            batch_images = batch_images / 255.0 # Normalize pixel values
            batch_volumes = np.array(batch_volumes, dtype="float32")
            
            yield batch_images, batch_volumes

# Create generators for training and validation
train_generator = data_generator(train_paths, train_volumes, BATCH_SIZE, is_training=True)
val_generator = data_generator(val_paths, val_volumes, BATCH_SIZE, is_training=False)

# --- 3. MODEL ARCHITECTURE ---
print("\nBuilding model...")
# Load the Xception model with pre-trained ImageNet weights, excluding the top layers
base_model = Xception(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Add a new regression head on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x) # Add L2 regularization
x = Dropout(0.3)(x) # Add a Dropout layer
x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x) # Add L2 regularization
predictions = Dense(1, activation='linear')(x) # Use 'linear' activation for regression

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# First stage: Freeze the convolutional layers and train only the head
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_INITIAL), loss='mean_squared_error', metrics=['mean_absolute_error'])
print("Training the regression head...")
model.fit(
    train_generator,
    steps_per_epoch=len(train_paths) // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=len(val_paths) // BATCH_SIZE,
    epochs=INITIAL_EPOCHS,
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), 
               ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)]
)


# Second stage: Fine-tuning
print("\nFine-tuning the model...")

# Unfreeze the last few convolutional blocks of the base model
for layer in base_model.layers:
    # A more selective approach to unfreeze only the last few blocks
    if "block13" in layer.name or "block14" in layer.name: # These are the last two blocks in Xception
        layer.trainable = True
    else:
        layer.trainable = False

# Re-compile the model with a lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE_FINE_TUNE),
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)

# Continue training with the fine-tuned model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_paths) // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=len(val_paths) // BATCH_SIZE,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
               ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)]
)

# --- 5. SAVING THE MODEL ---
print(f"\nTraining complete. Saving model to {MODEL_SAVE_PATH}")
model.save(MODEL_SAVE_PATH)
print("✅ Model saved successfully.")

# You can now unfreeze some layers for fine-tuning if needed
# or use this model for inference to estimate mangosteen volumes.
