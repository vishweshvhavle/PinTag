import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

class PinTagDetector:
    def __init__(self, debug=False, debug_dir="debug/", template_size=200,
                 inner_threshold=150, outer_threshold=180, 
                 red_threshold=100, green_threshold=150, 
                 min_aspect_ratio=0.5, max_aspect_ratio=6.0, min_roundness=0.5,
                 plot_normals=False) -> None:
        
        self.img = None
        self.debug = debug
        self.debug_dir = debug_dir
        self.orientation_matrix = None
        self.template_size = template_size
        self.centers = [(self.template_size//4, self.template_size//4), 
                        (3*self.template_size//4, self.template_size//4), 
                        (self.template_size//4, 3*self.template_size//4), 
                        (3*self.template_size//4, 3*self.template_size//4)]
        self.inner_radius = self.template_size // 20
        self.average_radius = (self.template_size // 10 + self.template_size // 5) // 2
        self.inner_threshold = inner_threshold
        self.outer_threshold = outer_threshold
        self.red_threshold = red_threshold
        self.green_threshold = green_threshold
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_roundness = min_roundness
        self.lab = None
        self.plot_normals = plot_normals
        self.original_img = None
        self.perspective_matrix = None
    
    def is_round(self, contour) -> bool:
        # Calculate the bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Aspect ratio check
        aspect_ratio = w / h if w > h else h / w
        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            return False
        
        # Roundness check (area to perimeter ratio)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        roundness = 4 * 3.14159 * area / (perimeter * perimeter)
        if roundness < self.min_roundness:
            return False

        return True

    def get_adjusted_center_order(self) -> (list, int): # type: ignore
        if np.array_equal(self.orientation_matrix, np.array([[1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1]])):
            return [self.centers[1], self.centers[3], self.centers[0], self.centers[2]], 1
        elif np.array_equal(self.orientation_matrix, np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 1, 0]])):
            return [self.centers[2], self.centers[0], self.centers[3], self.centers[1]], 3
        elif np.array_equal(self.orientation_matrix, np.array([[0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]])):
            return [self.centers[3], self.centers[2], self.centers[1], self.centers[0]], 2
        else:
            return self.centers, 0

    def find_red_circles_debug(self) -> list:
        a_channel = self.lab[:, :, 1]

        # Visualize LAB channels
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(self.lab[:, :, 0], cmap='gray')
        axes[0].set_title('L Channel')
        axes[1].imshow(a_channel, cmap='gray')
        axes[1].set_title('A Channel')
        axes[2].imshow(self.lab[:, :, 2], cmap='gray')
        axes[2].set_title('B Channel')
        plt.savefig(os.path.join(self.debug_dir, 'lab_channels.png'))
        plt.close()

        # Threshold the A channel to find red regions
        _, red_mask = cv2.threshold(a_channel, self.inner_threshold, 255, cv2.THRESH_BINARY)

        # Visualize red mask
        plt.imshow(red_mask, cmap='gray')
        plt.title('Red Mask')
        plt.savefig(os.path.join(self.debug_dir, 'red_mask.png'))
        plt.close()

        # Find contours in the red mask
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area (largest to smallest) and consider only the 10 biggest
        largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        # Filter out non-circular/elliptical contours
        filtered_contours = [contour for contour in largest_contours if self.is_round(contour)][:4]

        # Visualize contours
        contour_img = cv2.cvtColor(a_channel, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_img, filtered_contours, -1, (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
        plt.title('Detected Red Circles')
        plt.savefig(os.path.join(self.debug_dir, 'detected_red_circles.png'))
        plt.close()

        centroids = []
        for contour in filtered_contours:
            # Calculate the centroid of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append((cX, cY))
                cv2.circle(contour_img, (cX, cY), 5, (255, 0, 0), -1)
        
        plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
        plt.title('Detected Red Circles with Centroids')
        plt.savefig(os.path.join(self.debug_dir, 'red_circles_centroids.png'))
        plt.close()

        print(f"Detected centroids: {centroids}")
        return centroids
    
    def perspective_transform_debug(self, centroids) -> None:
        b_channel = self.lab[:,:,2]
        centroids = np.array(centroids, dtype="float32")

        # Compute the centroid of the points
        center = np.mean(centroids, axis=0)

        # Sort the points based on their positions relative to the centroid
        rect = np.zeros((4, 2), dtype="float32")
        centroids = sorted(centroids, key=lambda p: (np.arctan2(p[1] - center[1], p[0] - center[0])))

        # Order points in top-left, top-right, bottom-right, bottom-left order
        rect[0] = centroids[0]  # Top-left
        rect[1] = centroids[1]  # Top-right
        rect[2] = centroids[2]  # Bottom-right
        rect[3] = centroids[3]  # Bottom-left

        # Construct set of destination points for a square
        dst = np.array([
            [self.template_size//4, self.template_size//4], 
            [3*self.template_size//4, self.template_size//4], 
            [3*self.template_size//4, 3*self.template_size//4], 
            [self.template_size//4, 3*self.template_size//4]], dtype="float32")

        # Compute the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        self.img = cv2.warpPerspective(b_channel, M, (self.template_size, self.template_size))

        # Visualize the perspective transform
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        ax1.imshow(b_channel, cmap='gray')
        ax1.set_title('Original B Channel')
        for point in rect:
            ax1.plot(point[0], point[1], 'ro')
        ax2.imshow(self.img, cmap='gray')
        ax2.set_title('Perspective Transformed')
        for point in dst:
            ax2.plot(point[0], point[1], 'ro')
        plt.savefig(os.path.join(self.debug_dir, 'perspective_transform.png'))
        plt.close()

        print(f"Perspective transform applied. Input points: {rect}, Output points: {dst}")

        # Store the perspective transform matrix
        self.perspective_matrix = M
        self.img = cv2.warpPerspective(b_channel, M, (self.template_size, self.template_size))
    
    def decode_orientation_debug(self) -> None:
        self.orientation_matrix = np.zeros((4, 4), dtype=np.uint8)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.img, cmap='gray')
        ax.set_title('Orientation Decoding')

        for i, center in enumerate(self.centers):
            for j in range(4):
                angle = (j * np.pi / 2) + (np.pi / 4)
                x = int(center[0] + self.inner_radius * np.cos(angle))
                y = int(center[1] + self.inner_radius * np.sin(angle))

                if self.img[y, x] > self.red_threshold:
                    self.orientation_matrix[i, j] = 1
                    ax.plot(x, y, 'ro')
                else:
                    ax.plot(x, y, 'bo')

                print(f"Center {i}, Angle {j}: ({x}, {y}) - Value: {self.img[y, x]}")

        plt.savefig(os.path.join(self.debug_dir, 'orientation_decoding.png'))
        plt.close()

        print(f"Orientation matrix:\n{self.orientation_matrix}")
    
    def decode_green_sectors_debug(self) -> list:
        values = []
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.img, cmap='gray')
        ax.set_title('Green Sector Decoding - All Centers')

        adjusted_centers, rotation = self.get_adjusted_center_order()
        print(f"Adjusted centers: {adjusted_centers}, Rotation: {rotation}")
        for center_idx, center in enumerate(adjusted_centers):
            value = 0
            for i in range(8):
                angle = ((i + 2 * rotation) % 8 * np.pi / 4) + np.pi / 8
                angle_deg = np.degrees(angle)
                print(f"Angle: {angle_deg}")
                x = int(center[0] + self.average_radius * np.cos(angle))
                y = int(center[1] + self.average_radius * np.sin(angle))
                
                if self.img[y, x] > self.green_threshold:
                    value |= (1 << (7 - i))
                    ax.plot(x, y, 'go')
                else:
                    ax.plot(x, y, 'ro')
                
                if i == 0:
                    ax.plot(x, y, 'bo')
                    ax.text(x, y, f"Center {center_idx}", fontsize=12, color='white')

                print(f"Center {center_idx}, Sector {i}: ({x}, {y}) - Value: {self.img[y, x]}")

            values.append(value)
            print(f"Decoded value for center {center_idx}: {value}")

        plt.savefig(os.path.join(self.debug_dir, 'green_sector_decoding_all_centers.png'))
        plt.close()

        return values

    def find_red_circles(self) -> list:
        a_channel = self.lab[:, :, 1]

        # Threshold the A channel to find red regions
        _, red_mask = cv2.threshold(a_channel, self.inner_threshold, 255, cv2.THRESH_BINARY)

        # Find contours in the red mask
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area (largest to smallest) and consider only the 10 biggest
        largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        # Filter out non-circular/elliptical contours
        filtered_contours = [contour for contour in largest_contours if self.is_round(contour)][:4]

        centroids = []
        for contour in filtered_contours:
            # Calculate the centroid of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append((cX, cY))
        
        return centroids
    
    def perspective_transform(self, centroids) -> None:
        b_channel = self.lab[:,:,2]
        centroids = np.array(centroids, dtype="float32")

        # Compute the centroid of the points
        center = np.mean(centroids, axis=0)

        # Sort the points based on their positions relative to the centroid
        rect = np.zeros((4, 2), dtype="float32")
        centroids = sorted(centroids, key=lambda p: (np.arctan2(p[1] - center[1], p[0] - center[0])))

        # Order points in top-left, top-right, bottom-right, bottom-left order
        rect[0] = centroids[0]  # Top-left
        rect[1] = centroids[1]  # Top-right
        rect[2] = centroids[2]  # Bottom-right
        rect[3] = centroids[3]  # Bottom-left

        # Construct set of destination points for a square
        dst = np.array([
            [self.template_size//4, self.template_size//4], 
            [3*self.template_size//4, self.template_size//4], 
            [3*self.template_size//4, 3*self.template_size//4], 
            [self.template_size//4, 3*self.template_size//4]], dtype="float32")

        # Compute the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        self.img = cv2.warpPerspective(b_channel, M, (self.template_size, self.template_size))

        # Store the perspective transform matrix
        self.perspective_matrix = M
    
    def decode_orientation(self) -> None:
        self.orientation_matrix = np.zeros((4, 4), dtype=np.uint8)
        for i, center in enumerate(self.centers):
            for j in range(4):
                angle = (j * np.pi / 2) + (np.pi / 4)
                x = int(center[0] + self.inner_radius * np.cos(angle))
                y = int(center[1] + self.inner_radius * np.sin(angle))

                if self.img[y, x] > self.red_threshold:
                    self.orientation_matrix[i, j] = 1
    
    def decode_green_sectors(self) -> list:
        values = []
        adjusted_centers, rotation = self.get_adjusted_center_order()
        for center in adjusted_centers:
            value = 0
            for i in range(8):
                angle = ((i + 2 * rotation) % 8 * np.pi / 4) + np.pi / 8
                x = int(center[0] + self.average_radius * np.cos(angle))
                y = int(center[1] + self.average_radius * np.sin(angle))
                
                if self.img[y, x] > self.green_threshold:
                    value |= (1 << (7 - i))
            values.append(value)
        return values

    def plot_normal_vectors(self) -> None:
        inv_perspective = np.linalg.inv(self.perspective_matrix)

        def transform_point(point):
            homogeneous = np.append(point, 1)
            transformed = inv_perspective.dot(homogeneous)
            return transformed
        
        unnormalized_coords = [transform_point(center) for center in self.centers]
        
        if len(unnormalized_coords) >= 3:
            origin = np.mean(unnormalized_coords, axis=0)
            origin = origin / origin[2]
            origin_2d = tuple(map(int, origin[:2]))
            
            p1, p2, p3 = [np.array(coord) for coord in unnormalized_coords[:3]]
            normal = -1 * np.cross(p2[:3] - p1[:3], p3[:3] - p1[:3])

            # Calculate the end points for the X-axis
            x1 = np.append([self.template_size//4, self.template_size//4], 1)
            x2 = np.append([3*self.template_size//4, self.template_size//4], 1)
            x_mid = (np.array(inv_perspective.dot(x1)) + np.array(inv_perspective.dot(x2))) / 2
            x_end = tuple(map(int, x_mid[:2] / x_mid[2]))
            
            # Calculate the end points for the Y-axis
            y1 = np.append([self.template_size//4, self.template_size//4], 1)
            y2 = np.append([self.template_size//4, 3*self.template_size//4], 1)
            y_mid = (np.array(inv_perspective.dot(y1)) + np.array(inv_perspective.dot(y2))) / 2
            y_end = tuple(map(int, y_mid[:2] / y_mid[2]))

            # Calculate Z-axis end point
            z_end = origin[:2] + normal[:2] * ((np.linalg.norm(np.array(x_end) - origin[:2]) + np.linalg.norm(np.array(y_end) - origin[:2])) / 2) / np.linalg.norm(normal[:2])
            z_end = tuple(map(int, z_end))

            cv2.line(self.original_img, origin_2d, x_end, (0, 0, 255), 2) 
            cv2.line(self.original_img, origin_2d, y_end, (0, 255, 0), 2)
            cv2.line(self.original_img, origin_2d, z_end, (255, 0, 0), 2)

        cv2.imwrite('output_image.jpg', self.original_img) 

    def detect(self, img) -> list:
        if self.debug:
            print("Saving debug images to debug/ directory")

            # Convert the image to the LAB color space
            self.lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            print("Converted image to LAB color space")
            centroids = self.find_red_circles_debug()
            
            if len(centroids) != 4:
                print("Failed to detect 4 red circles")
                return None
            
            print("Red circles detected")
            
            self.perspective_transform_debug(centroids)
            print("Perspective transform done")

            self.decode_orientation_debug()
            print("Orientation decoded")

            # Decode values from the green sectors
            values = self.decode_green_sectors_debug()
            print("Green sectors decoded")
            
            # Last circle contains the checksum
            checksum = values.pop()

            # Verify checksum
            if sum(values) != checksum:
                print(f"Checksum verification failed: {sum(values)} != {checksum}")
                return None
            
            return values

        # Non-debug mode
        self.lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        centroids = self.find_red_circles()
        
        if len(centroids) != 4:
            return None
        
        self.perspective_transform(centroids)
        self.decode_orientation()

        # Decode values from the green sectors
        values = self.decode_green_sectors()
        
        # Last circle contains the checksum
        checksum = values.pop()

        # Verify checksum
        if sum(values) != checksum:
            print(f"Checksum verification failed: {sum(values)} != {checksum}")
            return None
        
        if self.plot_normals:
            self.original_img = img.copy()
            self.plot_normal_vectors()
        
        return values

def main() -> None:
    parser = argparse.ArgumentParser(description='Detect PinTag from Images')
    parser.add_argument('--input_image', type=str, required=True, help='Path to the IndiaTag image')
    parser.add_argument('--debug', action='store_true', help='Directory to save output images')
    parser.add_argument('--plot_normals', action='store_true', help='Plot normal vectors on the original image')
    args = parser.parse_args()
    if args.debug:
        print("Running in debug mode")
        os.makedirs("debug/", exist_ok=True)
        if args.plot_normals:
            detector = PinTagDetector(debug=True, debug_dir="debug/",plot_normals=args.plot_normals)
        else:
            detector = PinTagDetector(debug=True, debug_dir="debug/")
    else:
        if args.plot_normals:
            detector = PinTagDetector(plot_normals=args.plot_normals)
        else:
            detector = PinTagDetector()
    
    img = cv2.imread(args.input_image)
    result = detector.detect(img)

    if result:
        result = f"{result[0]:02d}{result[1]:02d}{result[2]:02d}"
        print(f"Detected PinTag values: {result}")
    else:
        print("Failed to detect PinTag")

if __name__ == "__main__":
    main()