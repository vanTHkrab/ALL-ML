import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.regularizers import l2
from tensorflow.keras.mixed_precision import Policy
from sklearn.model_selection import train_test_split
import cv2
from datetime import datetime
import logging
import gc

# --- GPU CONFIGURATION FOR RTX 3050 ---
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configure GPU memory growth for RTX 3050 (4GB VRAM)
def configure_gpu():
    """Configure GPU settings for RTX 3050"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth to avoid OOM errors
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            logger.info(f"✅ GPU detected: {gpus[0].name}")
            return True
        except RuntimeError as e:
            logger.error(f"❌ GPU configuration error: {e}")
            return False
    else:
        logger.warning("⚠️ No GPU detected, using CPU")
        return False


# Configure GPU
gpu_available = configure_gpu()

# Enable mixed precision training for RTX 3050 (supports Tensor Cores)
if gpu_available:
    try:
        policy = Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info(f"Mixed precision policy: {policy.name}")
    except Exception as e:
        logger.warning(f"Mixed precision not available: {e}")
        policy = Policy('float32')
        tf.keras.mixed_precision.set_global_policy(policy)

# --- 1. CONFIGURATION AND SETUP ---
# Paths
DATA_DIR = "./data/prepared/images"
VOLUME_FILES_DIR = "./data/prepared/volumes"
MODEL_SAVE_PATH = "./models/mangosteen_volume_model_rtx3050.h5"
CHECKPOINT_DIR = "./models/checkpoints"
LOG_DIR = "./logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

# Create directories if they don't exist
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Model parameters optimized for RTX 3050 (4GB VRAM)
IMG_SIZE = 224  # EfficientNet optimal size
BATCH_SIZE = 12  # Reduced for 4GB VRAM with safety margin
INITIAL_EPOCHS = 25
FINE_TUNE_EPOCHS = 35
LEARNING_RATE_INITIAL = 1e-3
LEARNING_RATE_FINE_TUNE = 1e-5
VALIDATION_SPLIT = 0.2


# --- 2. DATA LOADING AND PREPARATION ---

def load_data_from_folders(data_dir, volume_dir):
    """
    Loads images and their corresponding volume values with error handling.
    """
    image_paths = []
    volumes = []

    if not os.path.exists(data_dir):
        logger.error(f"Image data directory not found: {data_dir}")
        return [], []
    if not os.path.exists(volume_dir):
        logger.error(f"Volume data directory not found: {volume_dir}")
        return [], []

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    skipped_files = 0

    for filename in os.listdir(data_dir):
        if filename.lower().endswith(valid_extensions):
            base_filename = os.path.splitext(filename)[0]
            volume_filename = f"{base_filename}.txt"
            volume_filepath = os.path.join(volume_dir, volume_filename)

            if os.path.exists(volume_filepath):
                try:
                    with open(volume_filepath, 'r') as f:
                        volume_value = float(f.read().strip())

                    # Validate volume value
                    if volume_value <= 0 or volume_value > 10000:
                        logger.warning(f"Suspicious volume value {volume_value} for {filename}")
                        skipped_files += 1
                        continue

                    image_paths.append(os.path.join(data_dir, filename))
                    volumes.append(volume_value)

                except (ValueError, FileNotFoundError) as e:
                    logger.warning(f"Skipping {filename}: {e}")
                    skipped_files += 1
            else:
                logger.debug(f"No volume file for {filename}")
                skipped_files += 1

    logger.info(f"Loaded {len(image_paths)} samples, skipped {skipped_files} files")
    return image_paths, volumes


# Load data
logger.info("Loading data...")
image_paths, volumes = load_data_from_folders(DATA_DIR, VOLUME_FILES_DIR)

if not image_paths:
    logger.error("No matching image and volume data found. Please check your paths.")
    exit()

# Normalize volumes for better training stability
volumes = np.array(volumes)
volume_mean = np.mean(volumes)
volume_std = np.std(volumes)
volumes_normalized = (volumes - volume_mean) / volume_std

logger.info(f"Volume statistics - Mean: {volume_mean:.2f}, Std: {volume_std:.2f}")

# Split data
train_paths, val_paths, train_volumes, val_volumes = train_test_split(
    image_paths, volumes_normalized, test_size=VALIDATION_SPLIT, random_state=42
)

logger.info(f"Training samples: {len(train_paths)}, Validation samples: {len(val_paths)}")


# --- 3. OPTIMIZED DATA PIPELINE ---

@tf.function
def preprocess_image(image_path, volume, is_training=True):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])           # ✅ แทน reshape
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)

    if is_training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)
        image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))

    # คลิปไว้เฉย ๆ ก็ได้ แต่ปกติหลัง preprocess จะเป็น [-1,1]
    # ถ้าจะใช้ EfficientNet preprocess ภายหลัง ค่านี้ไม่ critical
    image = tf.clip_by_value(image, 0.0, 255.0)
    image.set_shape([IMG_SIZE, IMG_SIZE, 3])   # ✅ final static shape

    return image, volume



def create_dataset(paths, volumes, batch_size, is_training=True):
    """Create optimized tf.data pipeline"""
    paths = tf.constant(paths)
    volumes = tf.constant(volumes, dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, volumes))

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(1000, len(paths)))
        dataset = dataset.repeat()

    # Map preprocessing
    dataset = dataset.map(
        lambda x, y: preprocess_image(x, y, is_training),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# Create datasets
train_dataset = create_dataset(train_paths, train_volumes, BATCH_SIZE, is_training=True)
val_dataset = create_dataset(val_paths, val_volumes, BATCH_SIZE, is_training=False)


# --- 4. MODEL ARCHITECTURE ---
def create_model(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=input_shape, name='input_layer')
    x = tf.keras.layers.Lambda(
        tf.keras.applications.efficientnet.preprocess_input,
        name='effnet_preproc'
    )(inputs)

    try:
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape,  # ✅ เน้น 3-channel
            pooling=None
        )
        features = base_model(x)
        x = GlobalAveragePooling2D()(features)
        logger.info("✅ EfficientNetB0 loaded successfully")
    except Exception as e:
        logger.error(f"Error loading EfficientNetB0: {e}")
        logger.info("Using fallback CNN architecture")
        x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(128, 3, activation='relu')(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        base_model = None

    # ... (dense head เหมือนเดิม)

    if base_model is not None:
        # Get features from base model
        features = base_model.output

        # Global average pooling
        x = GlobalAveragePooling2D()(features)

    # Custom regression head
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.1)(x)

    # Output layer - ensure float32 for mixed precision
    outputs = Dense(1, activation='linear', dtype='float32', name='volume_output')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='mangosteen_volume_model')

    # Freeze base model initially if it exists
    if base_model is not None:
        base_model.trainable = False
        return model, base_model
    else:
        return model, None


# Create model
logger.info("Building model...")
try:
    model, base_model = create_model()
    logger.info("✅ Model created successfully")
    logger.info(f"Model has {model.count_params():,} parameters")
except Exception as e:
    logger.error(f"Error creating model: {e}")
    exit(1)

# --- 5. TRAINING CALLBACKS ---

callbacks = [
    # Save best model
    ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_DIR, 'best_model_{epoch:02d}_{val_loss:.4f}.h5'),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    ),

    # Early stopping
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),

    # Learning rate reduction
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),

    # TensorBoard logging
    TensorBoard(
        log_dir=LOG_DIR,
        histogram_freq=1,
        write_graph=True,
        write_images=False  # Disable to save memory
    )
]

# --- 6. TRAINING PHASES ---

# Calculate steps per epoch
steps_per_epoch = max(1, len(train_paths) // BATCH_SIZE)
validation_steps = max(1, len(val_paths) // BATCH_SIZE)

logger.info(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")

# Phase 1: Train head only
logger.info("Phase 1: Training model head...")
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE_INITIAL),
    loss='mse',
    metrics=['mae', 'mse']
)

try:
    history_head = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=INITIAL_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    logger.info("✅ Phase 1 completed successfully")
except Exception as e:
    logger.error(f"Error in Phase 1 training: {e}")
    # Save current model state
    model.save(MODEL_SAVE_PATH.replace('.h5', '_phase1_error.h5'))

# Clear memory
gc.collect()

# Phase 2: Fine-tuning (only if base_model exists)
if base_model is not None:
    logger.info("Phase 2: Fine-tuning model...")

    # Unfreeze top layers of base model
    base_model.trainable = True

    # Freeze bottom layers
    for layer in base_model.layers[:-30]:  # Keep more layers frozen for stability
        layer.trainable = False

    # Count trainable parameters
    trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    logger.info(f"Trainable parameters in fine-tuning: {trainable_count:,}")

    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_FINE_TUNE),
        loss='mse',
        metrics=['mae', 'mse']
    )

    try:
        history_finetune = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=FINE_TUNE_EPOCHS,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            initial_epoch=INITIAL_EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        logger.info("✅ Phase 2 completed successfully")
    except Exception as e:
        logger.error(f"Error in Phase 2 training: {e}")
else:
    logger.info("Skipping Phase 2: No base model to fine-tune")


# --- 7. SAVE FINAL MODEL ---

# Create prediction wrapper with denormalization
class VolumePredictor(tf.keras.Model):
    def __init__(self, base_model, mean, std):
        super().__init__()
        self.base_model = base_model
        self.mean = tf.constant(mean, dtype=tf.float32)
        self.std = tf.constant(std, dtype=tf.float32)

    @tf.function
    def call(self, inputs):
        normalized_pred = self.base_model(inputs)
        return normalized_pred * self.std + self.mean

    # Save model
    logger.info(f"Saving final model to {MODEL_SAVE_PATH}")
    try:
        model.save(MODEL_SAVE_PATH, save_format='h5')
        logger.info("✅ Model saved successfully")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        # Try saving weights only
        model.save_weights(MODEL_SAVE_PATH.replace('.h5', '_weights.h5'))
        logger.info("✅ Model weights saved successfully")

    # Save normalization parameters
    np.savez(
        MODEL_SAVE_PATH.replace('.h5', '_params.npz'),
        mean=volume_mean,
        std=volume_std
    )

    logger.info("✅ Training complete! Model saved successfully.")
    logger.info(f"📊 TensorBoard logs saved to: {LOG_DIR}")
    logger.info("Run 'tensorboard --logdir=./logs/fit' to view training metrics")


# --- 8. EVALUATION ---

def evaluate_model(model, dataset, steps, denormalize=True):
    """Evaluate model performance"""
    predictions = []
    actuals = []

    step_count = 0
    for images, volumes in dataset:
        if step_count >= steps:
            break

        try:
            preds = model.predict(images, verbose=0)
            predictions.extend(preds.flatten())
            actuals.extend(volumes.numpy())
            step_count += 1
        except Exception as e:
            logger.warning(f"Error in prediction step {step_count}: {e}")
            continue

    if not predictions:
        logger.error("No predictions were made")
        return {}

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    if denormalize:
        predictions = predictions * volume_std + volume_mean
        actuals = actuals * volume_std + volume_mean

    mae = np.mean(np.abs(predictions - actuals))
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)

    # Avoid division by zero in MAPE calculation
    non_zero_mask = actuals != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((actuals[non_zero_mask] - predictions[non_zero_mask]) / actuals[non_zero_mask])) * 100
    else:
        mape = float('inf')

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'num_samples': len(predictions)
    }


# Evaluate on validation set
logger.info("\nEvaluating model performance...")
try:
    metrics = evaluate_model(model, val_dataset, validation_steps)
    if metrics:
        for metric, value in metrics.items():
            if metric == 'num_samples':
                logger.info(f"{metric}: {value}")
            else:
                logger.info(f"{metric}: {value:.4f}")
    else:
        logger.warning("Could not evaluate model performance")
except Exception as e:
    logger.error(f"Error during evaluation: {e}")

logger.info("Script completed!")