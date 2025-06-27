import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Create output folder
os.makedirs("output/region_growing", exist_ok=True)

# Load or create a noisy image again (reusing the one from Task 1)
def create_noisy_image():
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (80, 80), 85, -1)
    cv2.rectangle(img, (120, 120), (170, 170), 170, -1)
    gauss = np.random.normal(0, 20, img.shape).astype(np.float32)
    noisy_img = img.astype(np.float32) + gauss
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

# Region growing algorithm
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

# Run Task 2
noisy_img = create_noisy_image()
seed_point = (50, 50)  # A point inside object 1
region_result = region_growing(noisy_img, seed=seed_point, threshold=15)

# Save outputs
cv2.imwrite("output/region_growing/noisy_image.png", noisy_img)
cv2.imwrite("output/region_growing/region_result.png", region_result)

# Display results
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(noisy_img, cmap='gray')
plt.title("Noisy Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(region_result, cmap='gray')
plt.title("Region Growing Result")
plt.axis('off')

plt.tight_layout()
plt.show()
