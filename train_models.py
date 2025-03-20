import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns



# Cấu hình tham số
CONFIG = {
    # Kích thước ảnh và training
    'IMAGE_SIZE': (128, 128),
    'BATCH_SIZE': 16,
    'EPOCHS': 50,
    'INITIAL_LR': 1e-4,

    # Augmentation
    'ROTATION_RANGE': 15,
    'BRIGHTNESS_RANGE': [0.8, 1.2],
    'HORIZONTAL_FLIP': True,
    'ZOOM_RANGE': 0.1,

    # Model
    'DROPOUT_RATE': 0.4,
    'L2_LAMBDA': 0.01,

    # Paths
    'BASE_DIR': 'data/split_dataset',
    'MODEL_PATH': 'models/best_model_4.h5',
    'LOG_DIR': 'logs'
}




class SafeDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, images_dir, labels_dir, batch_size=CONFIG['BATCH_SIZE'],
                 img_size=CONFIG['IMAGE_SIZE'], is_training=True, class_names=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.is_training = is_training
        self.class_names = class_names

        # Tạo danh sách các cặp (image, label)
        self.samples = []
        for action in os.listdir(images_dir):
            if not os.path.isdir(os.path.join(images_dir, action)):
                continue

            action_path = os.path.join(images_dir, action)
            for img_name in os.listdir(action_path):
                if not img_name.endswith(('.jpg', '.jpeg', '.png')):
                    continue

                img_path = os.path.join(action_path, img_name)
                label_path = os.path.join(
                    labels_dir,
                    action,
                    img_name.rsplit('.', 1)[0] + '.txt'
                )

                if os.path.exists(label_path):
                    self.samples.append((img_path, label_path, action))

        self.indexes = np.arange(len(self.samples))
        print(f"Found {len(self.samples)} samples in {images_dir}")

    def __len__(self):
        return len(self.samples) // self.batch_size

    def parse_yolo_label(self, label_content):
        """Parse YOLO format label và xử lý trường hợp nhiều bounding box."""
        lines = label_content.strip().split('\n')
        boxes = []
        class_ids = []

        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                boxes.append([x_center, y_center, width, height])
                class_ids.append(class_id)

        if boxes:
            areas = [w * h for _, _, w, h in boxes]
            max_idx = np.argmax(areas)
            return class_ids[max_idx], boxes[max_idx]

        return None, None

    def apply_augmentation(self, image, box=None):
        if not self.is_training:
            return image, box

        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-CONFIG['ROTATION_RANGE'], CONFIG['ROTATION_RANGE'])
            image = tf.image.rot90(image, k=int(angle/90))

        # Random brightness
        if np.random.random() > 0.5:
            image = tf.image.random_brightness(image, 0.2)
            image = tf.clip_by_value(image, 0, 1)

        # Random contrast
        if np.random.random() > 0.5:
            image = tf.image.random_contrast(image, 0.8, 1.2)
            image = tf.clip_by_value(image, 0, 1)

        # Random zoom - Sửa lại phần này
        if np.random.random() > 0.5:
            scale = np.random.uniform(1-CONFIG['ZOOM_RANGE'], 1+CONFIG['ZOOM_RANGE'])
            new_height = int(self.img_size[0] * scale)
            new_width = int(self.img_size[1] * scale)

            # Đảm bảo kích thước mới không nhỏ hơn 1
            new_height = max(1, new_height)
            new_width = max(1, new_width)

            image = tf.image.resize(image, (new_height, new_width))

            # Nếu scale > 1 (zoom in), cắt phần thừa
            # Nếu scale < 1 (zoom out), pad thêm
            image = tf.image.resize_with_crop_or_pad(
                image,
                self.img_size[0],
                self.img_size[1]
            )

        # Horizontal flip
        if CONFIG['HORIZONTAL_FLIP'] and np.random.random() > 0.5:
            image = tf.image.flip_left_right(image)
            if box is not None:
                x_center, y_center, width, height = box
                box = [1 - x_center, y_center, width, height]

        return image, box

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_samples = [self.samples[i] for i in batch_indexes]

        X = np.empty((self.batch_size, *self.img_size, 3))
        y = np.empty((self.batch_size,), dtype=int)

        for i, (img_path, label_path, action) in enumerate(batch_samples):
            # Load và xử lý ảnh
            img = load_img(img_path, target_size=self.img_size)
            img = img_to_array(img) / 255.0

            # Load và parse label
            with open(label_path, 'r') as f:
                label_content = f.read().strip()
            class_id, box = self.parse_yolo_label(label_content)

            if class_id is None:
                class_id = self.class_names.index(action)

            # Apply augmentation
            if self.is_training:
                img, box = self.apply_augmentation(img, box)

            X[i] = img
            y[i] = class_id

        # Convert to one-hot encoding
        y = tf.keras.utils.to_categorical(y, num_classes=len(self.class_names))

        return X, y

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indexes)

def create_improved_model(input_shape=(*CONFIG['IMAGE_SIZE'], 3), num_classes=6):
    model = Sequential([
        # First Conv Block
        Conv2D(64, (3, 3), padding='same', activation='relu',
               input_shape=input_shape,
               kernel_regularizer=tf.keras.regularizers.l2(CONFIG['L2_LAMBDA'])),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(CONFIG['DROPOUT_RATE']),

        # Second Conv Block
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(CONFIG['DROPOUT_RATE']),

        # Third Conv Block
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(CONFIG['DROPOUT_RATE']),

        # Fourth Conv Block
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(CONFIG['DROPOUT_RATE']),

        # Dense layers
        Flatten(),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=CONFIG['INITIAL_LR'],
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Accuracy per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_history128.png')
    plt.close()

def plot_confusion_matrix(model, test_generator, class_names):
    predictions = []
    true_labels = []

    for i in range(len(test_generator)):
        x, y = test_generator[i]
        pred = model.predict(x)
        predictions.extend(np.argmax(pred, axis=1))
        true_labels.extend(np.argmax(y, axis=1))

    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix128.png')
    plt.close()

def main():
    # Tạo thư mục cần thiết
    os.makedirs('models', exist_ok=True)
    os.makedirs(CONFIG['LOG_DIR'], exist_ok=True)

    # Định nghĩa classes
    actions = ['falling', 'jumping', 'running', 'sitting', 'standing', 'walking']

    # Tạo generators
    train_gen = SafeDataGenerator(
        os.path.join(CONFIG['BASE_DIR'], 'images/train'),
        os.path.join(CONFIG['BASE_DIR'], 'labels/train'),
        class_names=actions,
        is_training=True
    )

    val_gen = SafeDataGenerator(
        os.path.join(CONFIG['BASE_DIR'], 'images/val/val'),
        os.path.join(CONFIG['BASE_DIR'], 'labels/val'),
        class_names=actions,
        is_training=False
    )

    test_gen = SafeDataGenerator(
        os.path.join(CONFIG['BASE_DIR'], 'images/test/test'),
        os.path.join(CONFIG['BASE_DIR'], 'labels/test'),
        class_names=actions,
        is_training=False
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            CONFIG['MODEL_PATH'],
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=CONFIG['LOG_DIR'],
            histogram_freq=1
        )
    ]

    # Tạo và train model
    # Tạo và train model
    model = create_improved_model(num_classes=len(actions))
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG['EPOCHS'],
        callbacks=callbacks
    )

    # Plot training history
    plot_training_history(history)

    # Evaluate model
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"\nTest accuracy: {test_acc:.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(model, test_gen, actions)

if __name__ == "__main__":
    main()