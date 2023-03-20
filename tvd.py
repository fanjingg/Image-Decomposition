import numpy as np
import cv2

def tvd_denoising(image, alpha=0.3, beta=0.1, iterations=100):
    # Convert input image to float32
    image = image.astype(np.float32)

    # Initialize denoised image
    denoised_image = np.copy(image)

    # Calculate image gradients
    dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
    dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)

    # Perform TVD iterations
    for _ in range(iterations):
        # Update denoised image
        denoised_image = (1 - alpha) * denoised_image + alpha * cv2.blur(image, (3, 3))

        # Calculate denoised image gradients
        dx_denoised = cv2.Sobel(denoised_image, cv2.CV_32F, 1, 0, ksize=1)
        dy_denoised = cv2.Sobel(denoised_image, cv2.CV_32F, 0, 1, ksize=1)

        # Update image
        image = image + beta * (dx - dx_denoised) + beta * (dy - dy_denoised)

    return denoised_image

# Read input image
image_path = "000012.png"
input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Perform TVD denoising
denoised_image = tvd_denoising(input_image)

# Save denoised image
output_path = "path.jpg"
cv2.imwrite(output_path, denoised_image)
