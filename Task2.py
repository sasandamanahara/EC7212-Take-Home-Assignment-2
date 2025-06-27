import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Create output directory
os.makedirs("output/region_growing", exist_ok=True)


# Step 1: Create a synthetic grayscale image with two objects and background
def create_noisy_image():
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (80, 80), 85, -1)  # Object 1 (gray)
    cv2.rectangle(img, (120, 120), (170, 170), 170, -1)  # Object 2 (brighter)

    # Add Gaussian noise
    noise = np.random.normal(0, 20, img.shape).astype(np.float32)
    noisy_img = img.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img


# Step 2: Region Growing Function
def region_growing(image, seed, threshold=10):
    height, width = image.shape
    visited = np.zeros((height, width), dtype=bool)
    output = np.zeros((height, width), dtype=np.uint8)

    seed_value = image[seed]
    stack = [seed]

    while stack:
        x, y = stack.pop()
        if visited[x, y]:
            continue

        visited[x, y] = True
        output[x, y] = 255  # Mark as part of the region

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < height and 0 <= ny < width:
                    if not visited[nx, ny]:
                        neighbor_val = image[nx, ny]
                        if abs(int(neighbor_val) - int(seed_value)) <= threshold:
                            stack.append((nx, ny))

    return output


# Step 3: Run the segmentation
noisy_img = create_noisy_image()
seed_point = (50, 50)  # A point inside Object 1
threshold = 15  # Range for pixel similarity

segmentation_result = region_growing(noisy_img, seed_point, threshold)

# Step 4: Save the results
cv2.imwrite("output/region_growing/noisy_image.png", noisy_img)
cv2.imwrite("output/region_growing/region_result.png", segmentation_result)

# Step 5: Display the results
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(noisy_img, cmap='gray')
plt.title("Noisy Input Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmentation_result, cmap='gray')
plt.title("Region Growing Result")
plt.axis('off')

plt.tight_layout()
plt.savefig("output/region_growing/region_growing_summary.png")
plt.show()
