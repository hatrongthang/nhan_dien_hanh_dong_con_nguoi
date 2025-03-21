import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, LSTM, TimeDistributed
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
import cv2
from sklearn.utils.class_weight import compute_class_weight

# Định nghĩa các tham số
IMG_SIZE = (128, 128)  # Kích thước ảnh đầu vào
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 5
DATASET_PATH = "data/split_dataset"
NUM_CLASSES = 6
CLASS_LABELS = [
    "falling", "jumping", "running", "sitting", "standing", "walking"
]

def create_data_generators():
    """Tạo data generators cho training và validation."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    return train_datagen, val_datagen, test_datagen
def load_data(image_dir, label_dir):
    """Đọc ảnh và nhãn từ thư mục."""
    images = []
    labels = []

    for class_idx, class_name in enumerate(CLASS_LABELS):
        class_image_path = os.path.join(image_dir, class_name)
        class_label_path = os.path.join(label_dir, class_name)

        for img_name in os.listdir(class_image_path):
            img_path = os.path.join(class_image_path, img_name)
            label_path = os.path.join(
                class_label_path,
                os.path.splitext(img_name)[0] + ".txt"
            )

            # Đọc và xử lý ảnh
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            img = cv2.resize(img, IMG_SIZE)
            img = img / 255.0

            # Đọc nhãn YOLO format
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    line = f.readline().strip()
                    try:
                        # Lấy class_id từ định dạng YOLO
                        class_id = int(line.split()[0])
                        # Đảm bảo class_id nằm trong phạm vi hợp lệ
                        if class_id >= NUM_CLASSES:
                            print(f"Warning: Invalid class_id {class_id} in {label_path}")
                            continue
                        label = class_id
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Invalid label format in {label_path}: {e}")
                        continue
            else:
                label = class_idx

            images.append(img)
            labels.append(label)

    if not images:
        raise ValueError("No valid images found in the dataset")
        
    return np.array(images), np.array(labels)


def create_model():
    """Tạo mô hình CNN-LSTM với kiến trúc sâu hơn."""
    model = Sequential([
        # CNN layers
        TimeDistributed(
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            input_shape=(1, 128, 128, 3)
        ),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D(2, 2)),
        
        TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D(2, 2)),
        
        TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same')),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D(2, 2)),
        
        # Flatten và LSTM layers
        TimeDistributed(Flatten()),
        LSTM(128, return_sequences=True),
        LSTM(64, return_sequences=False),
        
        # Dense layers
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    return model


def create_callbacks():
    """Tạo các callback cho quá trình training."""
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.00001
        )
    ]
    return callbacks


def plot_training_history(history):
    """Vẽ biểu đồ loss và accuracy."""
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss per Epoch')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy per Epoch')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels):
    """Vẽ ma trận nhầm lẫn."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()


def main():
    # Tạo data generators
    train_datagen, val_datagen, test_datagen = create_data_generators()
    
    # Load dữ liệu
    print("Loading training data...")
    X_train, y_train = load_data(
        f"{DATASET_PATH}/images/train",
        f"{DATASET_PATH}/labels/train"
    )
    
    print("Loading validation data...")
    X_val, y_val = load_data(
        f"{DATASET_PATH}/images/val",
        f"{DATASET_PATH}/labels/val"
    )
    
    print("Loading test data...")
    X_test, y_test = load_data(
        f"{DATASET_PATH}/images/test",
        f"{DATASET_PATH}/labels/test"
    )
    
    # One-hot encoding
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_val = to_categorical(y_val, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)
    
    # Tính class weights
    y_train_classes = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_classes),
        y=y_train_classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    # Tạo và compile mô hình
    print("Creating model...")
    model = create_model()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        metrics=['accuracy']
    )
    
    # Tạo callbacks
    callbacks = create_callbacks()
    
    # Training
    print("Starting training...")
    history = model.fit(
        train_datagen.flow(
            X_train.reshape(-1, 1, 128, 128, 3),
            y_train,
            batch_size=BATCH_SIZE
        ),
        epochs=EPOCHS,
        validation_data=val_datagen.flow(
            X_val.reshape(-1, 1, 128, 128, 3),
            y_val,
            batch_size=BATCH_SIZE
        ),
        callbacks=callbacks,
        class_weight=class_weight_dict
    )
    
    # Đánh giá mô hình
    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(
        test_datagen.flow(
            X_test.reshape(-1, 1, 128, 128, 3),
            y_test,
            batch_size=BATCH_SIZE
        )
    )
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    # Vẽ biểu đồ
    print("Plotting training history...")
    plot_training_history(history)
    
    # Tạo ma trận nhầm lẫn
    print("Generating confusion matrix...")
    y_pred = model.predict(
        test_datagen.flow(
            X_test.reshape(-1, 1, 128, 128, 3),
            y_test,
            batch_size=BATCH_SIZE
        )
    )
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    plot_confusion_matrix(y_true, y_pred_classes, CLASS_LABELS)
    
    # In báo cáo phân loại
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=CLASS_LABELS))


if __name__ == "__main__":
    main()

