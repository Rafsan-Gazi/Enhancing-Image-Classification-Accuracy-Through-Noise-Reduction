import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Keep original copies of the train and test datasets for later use
X_train_original = X_train.copy()
X_test_original = X_test.copy()

# Function to add Gaussian noise to images
def add_gaussian_noise(images, mean=0, std=0.1):
    noise = np.random.normal(mean, std, images.shape)
    noisy_images = images + noise
    return np.clip(noisy_images, 0, 255)

# Function to add salt-and-pepper noise to images
def add_salt_pepper_noise(images, salt_prob=0.02, pepper_prob=0.02):
    noisy_images = images.copy()
    for img in noisy_images:
        num_salt = max(1, int(salt_prob * img.size))
        num_pepper = max(1, int(pepper_prob * img.size))
        coords_salt = [np.random.randint(0, max(1, dim), num_salt) for dim in img.shape]
        coords_pepper = [np.random.randint(0, max(1, dim), num_pepper) for dim in img.shape]
        img[tuple(coords_salt)] = 255
        img[tuple(coords_pepper)] = 0
    return noisy_images

# Function to add speckle noise to images
def add_speckle_noise(images):
    noise = np.random.randn(*images.shape)
    noisy_images = images + images * noise
    return np.clip(noisy_images, 0, 255)

# Function to combine multiple noise types into mixed noise
def add_mixed_noise(images):
    images = images.astype(np.float32)
    noisy_images = add_gaussian_noise(images, std=25)
    noisy_images = add_salt_pepper_noise(noisy_images, salt_prob=0.02, pepper_prob=0.02)
    noisy_images = add_speckle_noise(noisy_images)
    return noisy_images.astype(np.uint8)

# Add mixed noise to the test dataset
X_test_noisy = add_mixed_noise(X_test)

# Normalize and reshape datasets
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
X_test_noisy = X_test_noisy.reshape(-1, 28, 28, 1) / 255.0

# Convert labels to one-hot encoding
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# Define a simple CNN model for classification
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_one_hot, validation_data=(X_test, y_test_one_hot), epochs=10)

# Evaluate the model on clean test data
loss_clean, acc_clean = model.evaluate(X_test, y_test_one_hot)
print(f"Accuracy on clean test data: {acc_clean * 100:.2f}%")

# Evaluate the model on noisy test data
loss_noisy, acc_noisy = model.evaluate(X_test_noisy, y_test_one_hot)
print(f"Accuracy on noisy test data: {acc_noisy * 100:.2f}%")

# Define the autoencoder model for denoising
def autoencoder():
    input_img = Input(shape=(28, 28, 1))
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(encoded)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

# Initialize and train the autoencoder
autoencoder = autoencoder()
X_train_scaled = X_train_original.reshape(-1, 28, 28, 1) / 255.0
X_train_noisy_scaled = X_train.reshape(-1, 28, 28, 1)
autoencoder.fit(X_train_noisy_scaled, X_train_scaled, epochs=20, batch_size=128, validation_split=0.1)

# Define functions for different denoising techniques

# Median Filter
def apply_median_filter(images, kernel_size=3):
    return np.array([cv2.medianBlur((img.squeeze() * 255).astype(np.uint8), kernel_size) for img in images]).reshape(-1, 28, 28, 1)

# Gaussian Filter
def apply_gaussian_filter(images, kernel_size=3):
    return np.array([cv2.GaussianBlur((img.squeeze() * 255).astype(np.uint8), (kernel_size, kernel_size), 0) for img in images]).reshape(-1, 28, 28, 1)

# Bilateral Filter
def apply_bilateral_filter(images, diameter=9, sigma_color=75, sigma_space=75):
    return np.array([cv2.bilateralFilter((img.squeeze() * 255).astype(np.uint8), diameter, sigma_color, sigma_space) for img in images]).reshape(-1, 28, 28, 1)

# Non-Local Means Denoising
def apply_non_local_means(images, h=10):
    return np.array([cv2.fastNlMeansDenoising((img.squeeze() * 255).astype(np.uint8), None, h, 7, 21) for img in images]).reshape(-1, 28, 28, 1)

# Apply autoencoder for denoising
def apply_autoencoder(images, autoencoder_model):
    denoised_images = autoencoder_model.predict(images)
    return (denoised_images * 255).astype(np.uint8).reshape(-1, 28, 28, 1)

# Median followed by Gaussian Filter
def apply_median_then_gaussian(images):
    median_filtered = apply_median_filter(images)
    gaussian_filtered = apply_gaussian_filter(median_filtered)
    return gaussian_filtered

# Bilateral Filter followed by Autoencoder
def apply_bilateral_then_autoencoder(images, autoencoder_model):
    bilateral_filtered = apply_bilateral_filter(images)
    return apply_autoencoder(bilateral_filtered / 255.0, autoencoder_model)

# Median Filter followed by Non-Local Means Denoising
def apply_median_then_non_local_means(images):
    median_filtered = apply_median_filter(images)
    non_local_means_filtered = apply_non_local_means(median_filtered)
    return non_local_means_filtered

# Evaluate different denoising techniques
def evaluate_denoising_technique(denoise_func, images, model, y_test_one_hot, *args):
    if args:
        denoised_images = denoise_func(images, *args)
    else:
        denoised_images = denoise_func(images)
    denoised_images = denoised_images / 255.0 
    loss, accuracy = model.evaluate(denoised_images, y_test_one_hot, verbose=0)
    return accuracy

# Define a dictionary of denoising techniques
techniques = {
    "Median Filter": apply_median_filter,
    "Gaussian Filter": apply_gaussian_filter,
    "Bilateral Filter": apply_bilateral_filter,
    "Non-Local Means": apply_non_local_means,
    "Autoencoder": lambda imgs: apply_autoencoder(imgs / 255.0, autoencoder),
    "Median + Gaussian": apply_median_then_gaussian,
    "Bilateral + Autoencoder": lambda imgs: apply_bilateral_then_autoencoder(imgs, autoencoder),
    "Median + Non-Local Means": apply_median_then_non_local_means,
}

# Evaluate all techniques and store results
results = {}
for name, func in techniques.items():
    try:
        acc = evaluate_denoising_technique(func, X_test_noisy, model, y_test_one_hot)
        results[name] = acc
    except Exception as e:
        print(f"Error applying {name}: {e}")

# Create a DataFrame of the results for better visualization
results_df = pd.DataFrame(list(results.items()), columns=["Technique", "Accuracy"])
results_df["Accuracy (%)"] = results_df["Accuracy"] * 100
results_df.sort_values("Accuracy (%)", ascending=False, inplace=True)
print(results_df)

# Visualization and saving results as images
def save_images(images, title, filename, num_images=5):
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"/home/sk/Desktop/{filename}.png")
    plt.close()

# Save sample images from the original and noisy datasets
save_images(X_test_original[:5].reshape(-1, 28, 28, 1), "Clean", "clean_data")
save_images(X_test_noisy[:5], "Mixed Noise", "mixed_noise_data")

# Save images for each denoising technique
for name, func in techniques.items():
    try:
        if "Autoencoder" in name:
            denoised_images = func(X_test_noisy)
        else:
            denoised_images = func(X_test_noisy)
        save_images(denoised_images[:5], name, name.lower().replace(" ", "_") + "_denoised")
    except Exception as e:
        print(f"Error visualizing {name}: {e}")

