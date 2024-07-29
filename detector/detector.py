import argparse
import cv2
import numpy as np
from pdf2image import convert_from_path

import os

def save_lab_channels(img, output_dir):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Save each channel separately
    cv2.imwrite(os.path.join(output_dir, 'L_channel.png'), lab[:,:,0])
    cv2.imwrite(os.path.join(output_dir, 'A_channel.png'), lab[:,:,1])
    cv2.imwrite(os.path.join(output_dir, 'B_channel.png'), lab[:,:,2])
    
    # Save a colored version of the A channel for better visualization
    a_colored = cv2.applyColorMap(lab[:,:,1], cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(output_dir, 'A_channel_colored.png'), a_colored)

def find_red_circles(img, output_dir):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a_channel = lab[:,:,1]

    if output_dir:
        cv2.imwrite(os.path.join(output_dir, 'A_channel_before_threshold.png'), a_channel)
    
    # Threshold the A channel to find red regions
    _, red_mask = cv2.threshold(a_channel, 180, 255, cv2.THRESH_BINARY)
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, 'red_mask.png'), red_mask)
    
    # Find contours in the red mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    if output_dir:
        img_with_centroids = img.copy()
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append((cX, cY))
                cv2.circle(img_with_centroids, (cX, cY), 5, (0, 255, 0), -1)
                cv2.drawContours(img_with_centroids, [contour], 0, (255, 0, 0), 2)

        cv2.imwrite(os.path.join(output_dir, 'detected_centroids.png'), img_with_centroids)
    
        print(f"Found {len(centroids)} centroids")
        for i, centroid in enumerate(centroids):
            print(f"Centroid {i}: {centroid}")
    
    else:
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append((cX, cY))
    
    return centroids

def four_point_transform(image, pts, template_size=50):
    # Convert points to float32
    pts = np.array(pts, dtype="float32")
    
    # Compute the centroid of the points
    center = np.mean(pts, axis=0)
    
    # Sort the points based on their positions relative to the centroid
    rect = np.zeros((4, 2), dtype="float32")
    pts = sorted(pts, key=lambda p: (np.arctan2(p[1] - center[1], p[0] - center[0])))
    
    # Order points in top-left, top-right, bottom-right, bottom-left order
    rect[0] = pts[0]  # Top-left
    rect[1] = pts[1]  # Top-right
    rect[2] = pts[2]  # Bottom-right
    rect[3] = pts[3]  # Bottom-left
    
    # Define the template size for the square
    square_size = template_size

    # Calculate gutter size (10% of the square size on each side)
    gutter = int(square_size * 0.5)

    # New dimensions with gutters
    new_size = square_size + 2 * gutter

    # Construct set of destination points for a square with gutters
    dst = np.array([
        [gutter, gutter],
        [new_size - gutter - 1, gutter],
        [new_size - gutter - 1, new_size - gutter - 1],
        [gutter, new_size - gutter - 1]], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (new_size, new_size))

    return warped

def decode_green_sectors(img, center, inner_radius=10, outer_radius=20, output_dir=None):
    # Create a mask for the ring
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, outer_radius, 255, -1)
    cv2.circle(mask, center, inner_radius, 0, -1)
    
    # Convert to LAB color space and extract B channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    b_channel = lab[:,:,2]
    
    # Apply the mask to the B channel
    masked_b = cv2.bitwise_and(b_channel, b_channel, mask=mask)
    
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, f'masked_b_ring_{center}.png'), masked_b)
    
    # Initialize an output image for plotting
    plot_img = cv2.cvtColor(masked_b, cv2.COLOR_GRAY2BGR)

    threshold = 180
    
    value = 0
    for i in range(8):
        angle = (i * np.pi / 4) + np.pi / 8
        # Use the average radius for sampling
        avg_radius = (inner_radius + outer_radius) / 2
        x = int(center[0] + avg_radius * np.cos(angle))
        y = int(center[1] + avg_radius * np.sin(angle))
        if output_dir:
            print(f"Sampling point {i}: ({x}, {y}, {masked_b[y, x]})")
            cv2.circle(plot_img, (x, y), 5, (255, 0, 0), -1)
        
        # Use the threshold value at the specific point
        if masked_b[y, x] > threshold:
            value |= (1 << (7 - i))
    
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, f'result_with_samples{center}.png'), plot_img)
    
    return value

def decode_pose(img, output_dir=None):
    centers = [(25, 25), (75, 25), (25, 75), (75, 75)]

    # Convert to LAB color space and extract B channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    b_channel = lab[:,:,2]
    overall_matrix = np.zeros((4, 4))

    for i, center in enumerate(centers):
        # Create a mask for the inner circle
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, 100, 255, -1)
        masked_b = cv2.bitwise_and(b_channel, b_channel, mask=mask)
        
        if output_dir:
            # Save masked B channel for debugging
            cv2.imwrite(os.path.join(output_dir, f'masked_b_{i+1}.png'), masked_b)
        
            # Initialize an output image for plotting
            plot_img = cv2.cvtColor(masked_b, cv2.COLOR_GRAY2BGR)
        
        threshold = 100
        # Sample 4 points on the hemispheres at pi/4 intervals
        sampled_points = []
        for j in range(4):
            angle = (j * np.pi / 2) + (np.pi / 4)
            x = int(center[0] + 5 * np.cos(angle))
            y = int(center[1] + 5 * np.sin(angle))
            if output_dir:
                print(f"Sampling point {j+1}: ({x}, {y}, {masked_b[y, x]})")
                cv2.circle(plot_img, (x, y), 5, (255, 0, 0), -1)
            
            if masked_b[y, x] > threshold:
                overall_matrix[i][j] = 1
        
        if output_dir:
            # Save masked B channel for debugging
            cv2.imwrite(os.path.join(output_dir, f'sampled_red_{i+1}.png'), plot_img)
    
    if np.array_equal(overall_matrix, np.array([[0, 0,  1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0]])):
        warped = img
    # rotate left 90
    elif np.array_equal(overall_matrix, np.array([[1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1]])):
        warped = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # rotate upside down
    elif np.array_equal(overall_matrix, np.array([[0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]])):
        warped = cv2.rotate(img, cv2.ROTATE_180)
    # rotate right 90
    elif np.array_equal(overall_matrix, np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 1, 0]])):
        warped = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return warped, centers

def detect_indiatag(img, output_dir=None):
    centroids = find_red_circles(img, output_dir)
    
    if len(centroids) != 4:
        print(f"Expected 4 centroids, found {len(centroids)}")
        return None
    
    # Apply perspective transform
    warped = four_point_transform(img, np.array(centroids))
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, 'warped_tag.png'), warped)
    
    warped, centers = decode_pose(warped, output_dir)
    
    # Decode values from the green sectors
    values = []
    for i, center in enumerate(centers):
        value = decode_green_sectors(warped, center, output_dir=output_dir)
        values.append(value)
        if output_dir:
            print(f"Circle {i+1}: center = {center}, decoded value: {value}")
    
    # The last circle should be the checksum
    checksum = values.pop()
    
    # Verify checksum
    if sum(values) != checksum:
        print(f"Checksum verification failed: {sum(values)} != {checksum}")
        return None
    
    return values

def main():
    parser = argparse.ArgumentParser(description='Detect IndiaTag from PDF')
    parser.add_argument('--pdf_path', type=str, help='Path to the PDF file')
    parser.add_argument('--input_image', type=str, help='Path to the IndiaTag image')
    parser.add_argument('--output_dir', type=str, help='Directory to save output images')
    args = parser.parse_args()

    if args.input_image:
        img = cv2.imread(args.input_image, cv2.COLOR_RGB2BGR)

    # Convert PDF to image if available
    if args.pdf_path:
        images = convert_from_path(args.pdf_path)
        if not images:
            print("Failed to convert PDF to image")
            return

        # We assume the tag is on the first page
        img = np.array(images[0])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        # Save the original image
        cv2.imwrite(os.path.join(args.output_dir, 'original_image.png'), img)

        # Save LAB channel images
        save_lab_channels(img, args.output_dir)

        result = detect_indiatag(img, args.output_dir)
    else:
        result = detect_indiatag(img, None)

    if result:
        result = f"{result[0]:02d}{result[1]:02d}{result[2]:02d}"
        print(f"Detected IndiaTag values: {result}")
    else:
        print("Failed to detect IndiaTag")

if __name__ == "__main__":
    main()