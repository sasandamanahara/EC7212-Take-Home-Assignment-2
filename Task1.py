import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Create output folder
os.makedirs("output/otsu", exist_ok=True)

# Step 1: Create a simple image with 2 objects and background
def create_synthetic_image():
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (80, 80), 85, -1)       # Object 1
    cv2.rectangle(img, (120, 120), (170, 170), 170, -1)  # Object 2
    return img

# Step 2: Add Gaussian noise
def add_gaussian_noise(image, mean=0, sigma=20):
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_img = image.astype(np.float32) + gauss
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

# Step 3: Apply Otsu's Thresholding
def apply_otsu_threshold(noisy_img):
    threshold_value, otsu_img = cv2.threshold(noisy_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_img, threshold_value

# Run Task 1
original = create_synthetic_image()
noisy = add_gaussian_noise(original)
otsu_result, threshold_val = apply_otsu_threshold(noisy)

# Save outputs
cv2.imwrite("output/otsu/original_image.png", original)
cv2.imwrite("output/otsu/noisy_image.png", noisy)
cv2.imwrite("output/otsu/otsu_result.png", otsu_result)

# Display results
print(f"Otsu's Threshold Value: {threshold_val}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(original, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(noisy, cmap='gray')
plt.title("Noisy Image")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(otsu_result, cmap='gray')
plt.title("Otsu Result")
plt.axis('off')

plt.tight_layout()
plt.show()
