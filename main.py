import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Create output folders
os.makedirs("output/otsu", exist_ok=True)
os.makedirs("output/region_growing", exist_ok=True)

# -------- Task 1: Synthetic Image with Gaussian Noise + Otsu's Thresholding -------- #

# Step 1: Create a simple image with 2 objects and background
def create_synthetic_image():
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (80, 80), 85, -1)   # Object 1
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
    otsu_thresh_val, otsu_img = cv2.threshold(noisy_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_img, otsu_thresh_val

# -------- Task 2: Region Growing Segmentation -------- #

# Step 4: Region Growing Algorithm
def region_growing(img, seed, threshold=10):
    visited = np.zeros_like(img, dtype=bool)
    output = np.zeros_like(img, dtype=np.uint8)
    h, w = img.shape
    seed_val = img[seed]
    stack = [seed]

    while stack:
        x, y = stack.pop()
        if visited[x, y]:
            continue
        visited[x, y] = True
        output[x, y] = 255

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < h) and (0 <= ny < w):
                    if not visited[nx, ny] and abs(int(img[nx, ny]) - int(seed_val)) < threshold:
                        stack.append((nx, ny))

    return output

# -------- Main Execution -------- #

def main():
    # Task 1
    original = create_synthetic_image()
    noisy = add_gaussian_noise(original)
    otsu_result, threshold_val = apply_otsu_threshold(noisy)

    # Save images
    cv2.imwrite("output/otsu/original_image.png", original)
    cv2.imwrite("output/otsu/noisy_image.png", noisy)
    cv2.imwrite("output/otsu/otsu_result.png", otsu_result)

    print(f"Otsu's Threshold Value: {threshold_val}")

    # Task 2
    seed_point = (50, 50)  # Point inside Object 1
    region_result = region_growing(noisy, seed_point, threshold=15)

    # Save images
    cv2.imwrite("output/region_growing/noisy_image.png", noisy)
    cv2.imwrite("output/region_growing/region_result.png", region_result)

    # Display for verification
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    axs[0, 0].imshow(original, cmap='gray')
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(noisy, cmap='gray')
    axs[0, 1].set_title("Noisy Image")
    axs[0, 1].axis('off')

    axs[0, 2].imshow(otsu_result, cmap='gray')
    axs[0, 2].set_title("Otsu Result")
    axs[0, 2].axis('off')

    axs[1, 0].imshow(noisy, cmap='gray')
    axs[1, 0].set_title("Region Growing Input")
    axs[1, 0].axis('off')

    axs[1, 1].imshow(region_result, cmap='gray')
    axs[1, 1].set_title("Region Growing Output")
    axs[1, 1].axis('off')

    axs[1, 2].axis('off')  # Empty plot

    plt.tight_layout()
    plt.savefig("output/assignment2_summary.png")
    plt.show()

if __name__ == "__main__":
    main()
