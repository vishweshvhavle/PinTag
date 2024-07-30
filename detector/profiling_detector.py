import argparse
import cv2
import numpy as np
from pdf2image import convert_from_path
import os
import timeit

def save_lab_channels(img, output_dir):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    cv2.imwrite(os.path.join(output_dir, 'L_channel.png'), lab[:,:,0])
    cv2.imwrite(os.path.join(output_dir, 'A_channel.png'), lab[:,:,1])
    cv2.imwrite(os.path.join(output_dir, 'B_channel.png'), lab[:,:,2])
    a_colored = cv2.applyColorMap(lab[:,:,1], cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(output_dir, 'A_channel_colored.png'), a_colored)

def is_circular_or_elliptical(contour, min_aspect_ratio=0.5, max_aspect_ratio=2.0, min_roundness=0.7):
    # Calculate the bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    
    # Aspect ratio check
    aspect_ratio = w / h if w > h else h / w
    if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
        return False
    
    # Roundness check (area to perimeter ratio)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    roundness = 4 * 3.14159 * area / (perimeter * perimeter)
    if roundness < min_roundness:
        return False

    return True

def find_red_circles(img, output_dir=None):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]

    if output_dir:
        cv2.imwrite(os.path.join(output_dir, 'A_channel_before_threshold.png'), a_channel)
    
    # Threshold the A channel to find red regions
    _, red_mask = cv2.threshold(a_channel, 150, 255, cv2.THRESH_BINARY)
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, 'red_mask.png'), red_mask)
    
    # Find contours in the red mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area (largest to smallest) and consider only the 10 biggest
    largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Filter out non-circular/elliptical contours
    filtered_contours = [contour for contour in largest_contours if is_circular_or_elliptical(contour)]
    
    # Sort filtered contours by area (largest to smallest) and keep only the 4 largest
    filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)[:4]
    
    centroids = []
    if output_dir:
        img_with_centroids = img.copy()
        for contour in filtered_contours:
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
        for contour in filtered_contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append((cX, cY))
    
    return centroids

def four_point_transform(image, pts, template_size=50):
    pts = np.array(pts, dtype="float32")
    center = np.mean(pts, axis=0)
    rect = np.zeros((4, 2), dtype="float32")
    pts = sorted(pts, key=lambda p: (np.arctan2(p[1] - center[1], p[0] - center[0])))
    rect[0] = pts[0]
    rect[1] = pts[1]
    rect[2] = pts[2]
    rect[3] = pts[3]
    square_size = template_size
    gutter = int(square_size * 0.5)
    new_size = square_size + 2 * gutter
    dst = np.array([
        [gutter, gutter],
        [new_size - gutter - 1, gutter],
        [new_size - gutter - 1, new_size - gutter - 1],
        [gutter, new_size - gutter - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (new_size, new_size))
    return warped

def decode_green_sectors(img, center, inner_radius=10, outer_radius=20, output_dir=None):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, outer_radius, 255, -1)
    cv2.circle(mask, center, inner_radius, 0, -1)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    b_channel = lab[:,:,2]
    masked_b = cv2.bitwise_and(b_channel, b_channel, mask=mask)
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, f'masked_b_ring_{center}.png'), masked_b)
    plot_img = cv2.cvtColor(masked_b, cv2.COLOR_GRAY2BGR)
    threshold = 180
    value = 0
    for i in range(8):
        angle = (i * np.pi / 4) + np.pi / 8
        avg_radius = (inner_radius + outer_radius) / 2
        x = int(center[0] + avg_radius * np.cos(angle))
        y = int(center[1] + avg_radius * np.sin(angle))
        if output_dir:
            print(f"Sampling point {i}: ({x}, {y}, {masked_b[y, x]})")
            cv2.circle(plot_img, (x, y), 5, (255, 0, 0), -1)
        if masked_b[y, x] > threshold:
            value |= (1 << (7 - i))
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, f'result_with_samples{center}.png'), plot_img)
    return value

def decode_pose(img, output_dir=None):
    centers = [(25, 25), (75, 25), (25, 75), (75, 75)]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    b_channel = lab[:,:,2]
    overall_matrix = np.zeros((4, 4))
    for i, center in enumerate(centers):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, 100, 255, -1)
        masked_b = cv2.bitwise_and(b_channel, b_channel, mask=mask)
        if output_dir:
            cv2.imwrite(os.path.join(output_dir, f'masked_b_{i+1}.png'), masked_b)
            plot_img = cv2.cvtColor(masked_b, cv2.COLOR_GRAY2BGR)
        threshold = 100
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
            cv2.imwrite(os.path.join(output_dir, f'sampled_red_{i+1}.png'), plot_img)
    if np.array_equal(overall_matrix, np.array([[0, 0,  1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0]])):
        warped = img
    elif np.array_equal(overall_matrix, np.array([[1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1]])):
        warped = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif np.array_equal(overall_matrix, np.array([[0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]])):
        warped = cv2.rotate(img, cv2.ROTATE_180)
    elif np.array_equal(overall_matrix, np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 1, 0]])):
        warped = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return warped, centers

def detect_indiatag(img, output_dir=None):
    centroids = find_red_circles(img, output_dir)
    if len(centroids) != 4:
        print(f"Expected 4 centroids, found {len(centroids)}")
        return None
    warped = four_point_transform(img, np.array(centroids))
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, 'warped_tag.png'), warped)
    warped, centers = decode_pose(warped, output_dir)
    values = []
    for i, center in enumerate(centers):
        value = decode_green_sectors(warped, center, output_dir=output_dir)
        values.append(value)
        if output_dir:
            print(f"Circle {i+1}: center = {center}, decoded value: {value}")
    checksum = values.pop()
    if sum(values) != checksum:
        print(f"Checksum verification failed: {sum(values)} != {checksum}")
        return None
    return values

def profile_function(func, *args, **kwargs):
    timer = timeit.Timer(lambda: func(*args, **kwargs))
    times = timer.repeat(repeat=10, number=1)
    print(f"Function '{func.__name__}' execution times: {times}")
    print(f"Average time: {sum(times) / len(times):.5f} seconds\n")

def main():
    parser = argparse.ArgumentParser(description='Detect IndiaTag from PDF')
    parser.add_argument('--pdf_path', type=str, help='Path to the PDF file')
    parser.add_argument('--input_image', type=str, help='Path to the IndiaTag image')
    parser.add_argument('--output_dir', type=str, help='Directory to save output images')
    args = parser.parse_args()

    if args.input_image:
        img = cv2.imread(args.input_image, cv2.COLOR_RGB2BGR)

    if args.pdf_path:
        images = convert_from_path(args.pdf_path)
        if not images:
            print("Failed to convert PDF to image")
            return
        img = np.array(images[0])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(args.output_dir, 'original_image.png'), img)
        save_lab_channels(img, args.output_dir)
        result = detect_indiatag(img, args.output_dir)
    else:
        result = profile_function(detect_indiatag, img, None)

    if result:
        result = f"{result[0]:02d}{result[1]:02d}{result[2]:02d}"
        print(f"Detected IndiaTag values: {result}")
    else:
        print("Failed to detect IndiaTag")

if __name__ == "__main__":
    main()
