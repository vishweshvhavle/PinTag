import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
class SquareFinder:
    @staticmethod
    def is_bounded_by_black(gray, contour, threshold=10):
        # Create a mask of the contour
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)

        # Dilate the mask to check the boundary
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        boundary_mask = cv2.subtract(dilated_mask, mask)

        # Check if all boundary pixels are black (or close to black)
        boundary_pixels = gray[boundary_mask == 255]
        return np.all(boundary_pixels < threshold)

    @staticmethod
    def find_squares(gray, limit_cosine=0.9, min_area=100, max_error=0.025):
        squares = []
        
        # Find contours
        contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Approximate contour with accuracy proportional to the contour perimeter
            epsilon = max_error * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Square contours should have 4 vertices, relatively large area, and be convex
            if len(approx) == 4 and cv2.contourArea(approx) > min_area and cv2.isContourConvex(approx):
                max_cosine = 0
                
                # Find the maximum cosine of the angle between joint edges
                for j in range(2, 5):
                    cosine = abs(SquareFinder.angle_corner_points_cos(
                        approx[j % 4][0], approx[j-2][0], approx[j-1][0]))
                    max_cosine = max(max_cosine, cosine)
                
                if max_cosine < limit_cosine:
                    if SquareFinder.is_bounded_by_black(gray, contour):
                        squares.append(approx)
        
        return squares

    @staticmethod
    def draw_quads(mat, quads):
        cv2.drawContours(mat, quads, -1, (255, 0, 255), 2)

    @staticmethod
    def angle_corner_points_cos(b, c, a):
        dx1 = float(b[0] - a[0])
        dy1 = float(b[1] - a[1])
        dx2 = float(c[0] - a[0])
        dy2 = float(c[1] - a[1])
        
        magnitude1 = dx1*dx1 + dy1*dy1
        magnitude2 = dx2*dx2 + dy2*dy2
        dot_product = dx1*dx2 + dy1*dy2
        
        # Avoid division by zero and square root of negative numbers
        if magnitude1 == 0 or magnitude2 == 0:
            return 0  # or another appropriate value
        
        cos_value = dot_product / (np.sqrt(magnitude1) * np.sqrt(magnitude2))
        
        # Clamp the value to [-1, 1] to avoid domain errors in arccos
        return max(-1, min(1, cos_value))
    
class PinTagDetector:
    def __init__(self, debug=False, debug_dir="debug/", template_size=200, max_ids=1,
                 inner_threshold=150, outer_threshold=180, 
                 red_threshold=100, green_threshold=150, 
                 min_aspect_ratio=0.5, max_aspect_ratio=6.0, min_roundness=0.5,
                 plot_normals=False) -> None:
        
        self.img = None
        self.debug = debug
        self.debug_dir = debug_dir
        self.orientation_matrix = None
        self.max_ids = max_ids
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
        self.perspective_matrices = {}
        self.centroid_groups = {}
        self.centroid_group_ids = []
    
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
        if perimeter == 0:
            return False
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
        # L Channel
        axes[0].imshow(self.lab[:, :, 0], cmap='gray')
        # axes[0].set_title('L Channel')
        axes[0].axis('off')

        # A Channel
        axes[1].imshow(a_channel, cmap='gray')
        # axes[1].set_title('A Channel')
        axes[1].axis('off')

        # B Channel
        axes[2].imshow(self.lab[:, :, 2], cmap='gray')
        # axes[2].set_title('B Channel')
        axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.debug_dir, 'lab_channels.png'))
        plt.close()

        _, red_mask = cv2.threshold(a_channel, self.inner_threshold, 255, cv2.THRESH_BINARY)

        # Visualize red mask
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(red_mask, cmap='gray')
        # plt.title('Red Mask')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.debug_dir, 'red_mask.png'))
        plt.close()

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
        filtered_contours = [contour for contour in largest_contours if self.is_round(contour)]

        # Visualize contours
        contour_img = cv2.cvtColor(a_channel, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_img, filtered_contours, -1, (0, 255, 0), 2)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
        # plt.title('Detected Red Circles')
        plt.axis('off')
        plt.tight_layout()
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
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
        # plt.title('Detected Red Circles with Centroids')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.debug_dir, 'red_circles_centroids.png'))
        plt.close()

        print(f"Detected centroids: {centroids}")
        return centroids
    
    def perspective_transform_debug(self, centroids, centroids_id) -> None:
        b_channel = self.lab[:,:,2]
        centroids = np.array(centroids, dtype="float32")

        center = np.mean(centroids, axis=0)
        rect = np.zeros((4, 2), dtype="float32")
        centroids = sorted(centroids, key=lambda p: (np.arctan2(p[1] - center[1], p[0] - center[0])))

        rect[0] = centroids[0]  # Top-left
        rect[1] = centroids[1]  # Top-right
        rect[2] = centroids[2]  # Bottom-right
        rect[3] = centroids[3]  # Bottom-left

        dst = np.array([
            [self.template_size//4, self.template_size//4], 
            [3*self.template_size//4, self.template_size//4], 
            [3*self.template_size//4, 3*self.template_size//4], 
            [self.template_size//4, 3*self.template_size//4]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        self.img = cv2.warpPerspective(b_channel, M, (self.template_size, self.template_size))

        # Visualize the perspective transform
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        ax1.axis('off')
        ax1.imshow(b_channel, cmap='gray')
        # ax1.set_title('Original B Channel')
        for point in rect:
            ax1.plot(point[0], point[1], 'ro')
        ax2.axis('off')
        ax2.imshow(self.img, cmap='gray')
        # ax2.set_title('Perspective Transformed')
        for point in dst:
            ax2.plot(point[0], point[1], 'ro')
        plt.tight_layout()
        plt.savefig(os.path.join(self.debug_dir, f'perspective_transform_{centroids_id}.png'))
        plt.close()

        print(f"Perspective transform applied. Input points: {rect}, Output points: {dst}, Centroids ID: {centroids_id}")

        # Store the perspective transform matrix
        self.perspective_matrices[centroids_id] = M
        self.img = cv2.warpPerspective(b_channel, M, (self.template_size, self.template_size))
    
    def decode_orientation_debug(self, centroid_group_id) -> None:
        self.orientation_matrix = np.zeros((4, 4), dtype=np.uint8)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.img, cmap='gray')
        # ax.set_title(f'Orientation Decoding - Centroid Group ID: {centroid_group_id}')

        all_sector_values = []

        for i, center in enumerate(self.centers):
            sector_values = []
            for j in range(4):
                angle = (j * np.pi / 2) + (np.pi / 4)
                x = int(center[0] + self.inner_radius * np.cos(angle))
                y = int(center[1] + self.inner_radius * np.sin(angle))
                sector_values.append(self.img[y, x])
            all_sector_values.extend(sector_values)

        # Dynamic thresholding to decode the values
        self.red_threshold = self.dynamic_threshold_decode(all_sector_values)

        for i, center in enumerate(self.centers):
            for j in range(4):
                angle = (j * np.pi / 2) + (np.pi / 4)
                x = int(center[0] + self.inner_radius * np.cos(angle))
                y = int(center[1] + self.inner_radius * np.sin(angle))
                if self.img[y, x] > self.red_threshold:
                    self.orientation_matrix[i, j] = 1
                    ax.plot(x, y, 'go')
                else:
                    ax.plot(x, y, 'ro')

                print(f"Center {i}, Angle {j}: ({x}, {y}) - Value: {self.img[y, x]}")

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.debug_dir, f'orientation_decoding_{centroid_group_id}.png'))
        plt.close()

        print(f"Orientation matrix:\n{self.orientation_matrix}")
    
    def dynamic_threshold_decode(self, values) -> float:
        groups = []
        for v in values:
            for group in groups:
                if abs(v - sum(group) / len(group)) <= 15:
                    group.append(v)
                    break
            else:
                groups.append([v])
        groups.sort(key=len, reverse=True)
        threshold = (sum(groups[0]) / len(groups[0]) + sum(groups[1]) / len(groups[1])) / 2
        return threshold

    def decode_green_sectors_debug(self, centroid_group_id) -> list:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.img, cmap='gray')

        adjusted_centers, rotation = self.get_adjusted_center_order()
        all_sector_values = []

        # Collect all sector values first
        for center_idx, center in enumerate(adjusted_centers):
            sector_values = []
            for i in range(8):
                angle = ((i + 2 * rotation) % 8 * np.pi / 4) + np.pi / 8
                x = int(center[0] + self.average_radius * np.cos(angle))
                y = int(center[1] + self.average_radius * np.sin(angle))
                sector_values.append(self.img[y, x])
                print(f"Center {center_idx}, Sector {i}: ({x}, {y}) - Value: {self.img[y, x]}")
            all_sector_values.extend(sector_values)

        # Dynamic thresholding to decode the values
        global_threshold = self.dynamic_threshold_decode(all_sector_values)
        print(f"Global threshold: {global_threshold}")

        decoded_values = []

        for center_idx, center in enumerate(adjusted_centers):
            sector_values = []
            for i in range(8):
                angle = ((i + 2 * rotation) % 8 * np.pi / 4) + np.pi / 8
                x = int(center[0] + self.average_radius * np.cos(angle))
                y = int(center[1] + self.average_radius * np.sin(angle))    
                sector_values.append(self.img[y, x])
                ax.plot(x, y, 'go' if self.img[y, x] > global_threshold else 'ro')

                if i == 0:
                    ax.text(center[0]-12, center[1]-3, f"Center {center_idx}", fontsize=20, color='red')
                    
                    # Draw a radius to indicate the first marker
                    start_x = center[0] + self.inner_radius * 2 * (np.cos(angle-np.pi/8))
                    start_y = center[1] + self.inner_radius * 2 * (np.sin(angle-np.pi/8))
                    end_x = int(center[0] + self.inner_radius * 4 * (np.cos(angle-np.pi/8)))
                    end_y = int(center[1] + self.inner_radius * 4 * (np.sin(angle-np.pi/8)))
                    ax.plot([start_x, end_x], [start_y, end_y], 'b-', linewidth=2)
            
            decoded_value = int(''.join(['1' if v > global_threshold else '0' for v in sector_values]), 2)
            decoded_values.append(decoded_value)

            print(f"Decoded value for center {center_idx}: {decoded_value}")

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.debug_dir, f'green_sectors_decoding_{centroid_group_id}.png'))
        plt.close()

        return decoded_values

    def find_red_circles(self) -> list:
        a_channel = self.lab[:, :, 1]

        _, red_mask = cv2.threshold(a_channel, self.inner_threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

        # Filter out non-circular/elliptical contours
        filtered_contours = [contour for contour in largest_contours if self.is_round(contour)]

        centroids = []
        for contour in filtered_contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append((cX, cY))
        
        return centroids
    
    def perspective_transform(self, centroids, centroid_group_id) -> None:
        b_channel = self.lab[:,:,2]
        centroids = np.array(centroids, dtype="float32")

        center = np.mean(centroids, axis=0)
        rect = np.zeros((4, 2), dtype="float32")
        centroids = sorted(centroids, key=lambda p: (np.arctan2(p[1] - center[1], p[0] - center[0])))

        rect[0] = centroids[0]  # Top-left
        rect[1] = centroids[1]  # Top-right
        rect[2] = centroids[2]  # Bottom-right
        rect[3] = centroids[3]  # Bottom-left

        dst = np.array([
            [self.template_size//4, self.template_size//4], 
            [3*self.template_size//4, self.template_size//4], 
            [3*self.template_size//4, 3*self.template_size//4], 
            [self.template_size//4, 3*self.template_size//4]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        self.img = cv2.warpPerspective(b_channel, M, (self.template_size, self.template_size))
        self.perspective_matrices[centroid_group_id] = M
    
    def decode_orientation(self) -> None:
        self.orientation_matrix = np.zeros((4, 4), dtype=np.uint8)

        all_sector_values = []

        for i, center in enumerate(self.centers):
            sector_values = []
            for j in range(4):
                angle = (j * np.pi / 2) + (np.pi / 4)
                x = int(center[0] + self.inner_radius * np.cos(angle))
                y = int(center[1] + self.inner_radius * np.sin(angle))
                sector_values.append(self.img[y, x])
            all_sector_values.extend(sector_values)

        # Dynamic thresholding to decode the values
        self.red_threshold = self.dynamic_threshold_decode(all_sector_values)

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
        all_sector_values = []

        for center in adjusted_centers:
            for i in range(8):
                angle = ((i + 2 * rotation) % 8 * np.pi / 4) + np.pi / 8
                x = int(center[0] + self.average_radius * np.cos(angle))
                y = int(center[1] + self.average_radius * np.sin(angle))
                all_sector_values.append(self.img[y, x])
        self.green_threshold = self.dynamic_threshold_decode(all_sector_values)

        for center in adjusted_centers:
            sector_values = []
            for i in range(8):
                angle = ((i + 2 * rotation) % 8 * np.pi / 4) + np.pi / 8
                x = int(center[0] + self.average_radius * np.cos(angle))
                y = int(center[1] + self.average_radius * np.sin(angle))
                sector_values.append(self.img[y, x])
            
            decoded_value = int(''.join(['1' if v > self.green_threshold else '0' for v in sector_values]), 2)
            values.append(decoded_value)

        return values

    def plot_normal_vectors(self, output_dir) -> None:
        def transform_point(point):
            homogeneous = np.append(point, 1)
            transformed = inv_perspective.dot(homogeneous)
            return transformed
        
        for centroid_group_id in self.centroid_group_ids:
            inv_perspective = np.linalg.inv(self.perspective_matrices[centroid_group_id])
            
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

                cv2.line(self.original_img, origin_2d, x_end, (0, 0, 255), 20) 
                cv2.line(self.original_img, origin_2d, y_end, (0, 255, 0), 20)
                cv2.line(self.original_img, origin_2d, z_end, (255, 0, 0), 20)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB))
        # ax.set_title(f'Plotted Normal Vectors')

        output_path = os.path.join(output_dir, 'normal_vectors.png')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def group_centroids(self, centroids, centroid_mask=True) -> None:
        l_channel = self.lab[:,:,0]
        _, binary = cv2.threshold(l_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        if centroid_mask:
            threshold_width = np.linalg.norm(np.array(centroids[1]) - np.array(centroids[0]))
            centroid_mask = np.zeros_like(binary)
            for centroid in centroids:
                cv2.circle(centroid_mask, (int(centroid[0]), int(centroid[1])), int(threshold_width), 255, -1)
            binary = cv2.bitwise_and(binary, centroid_mask)

        quads = SquareFinder.find_squares(binary)
        quads = sorted(quads, key=cv2.contourArea, reverse=True)
        
        self.centroid_groups = {}
        self.centroid_group_ids = []
        group_id = 0
        filtered_quads = []
        
        for quad in quads:
            mask = np.zeros(l_channel.shape, dtype=np.uint8)
            cv2.drawContours(mask, [quad], 0, (255), -1)
            
            # Find centroids that fall within this quad
            group_centroids = [centroid for centroid in centroids 
                            if mask[int(centroid[1]), int(centroid[0])] == 255]
            
            # If we found exactly 4 centroids in this quad, it's a valid group
            if len(group_centroids) == 4:
                self.centroid_groups[group_id] = group_centroids
                self.centroid_group_ids.append(group_id)
                group_id += 1
                centroids = [c for c in centroids if c not in group_centroids]
            
            filtered_quads.append(quad)
            if group_id >= self.max_ids or not centroids:
                break
        
        if self.debug:
            self.group_centroids_debug(filtered_quads, binary)

    def group_centroids_debug(self, quads, binary):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(binary, cmap='gray')
        # ax.set_title('Binary Image with Detected Quads and Centroids')

        for quad in quads:
            quad = quad.reshape(-1, 2)
            quad = np.vstack((quad, quad[0]))  # Close the polygon
            ax.plot(quad[:, 0], quad[:, 1], color='lime', linewidth=2)

        for group_id, centroids in self.centroid_groups.items():
            x_coords, y_coords = zip(*centroids)
            ax.scatter(x_coords, y_coords, color='cyan', s=150, zorder=5)

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.debug_dir, 'detected_tags.png'))
        plt.close(fig)

    def detect(self, img) -> list:
        if self.debug:
            print("Saving debug images to debug/ directory")

            self.lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            print("Converted image to LAB color space")
            centroids = self.find_red_circles_debug()
            
            if len(centroids) < 4:
                return None
            
            print("Red circles detected")

            self.group_centroids(centroids)
            
            results = []
            for centroid_group_id in self.centroid_group_ids:
                self.perspective_transform_debug(self.centroid_groups[centroid_group_id], centroid_group_id)
                print("Perspective transform done")

                self.decode_orientation_debug(centroid_group_id)
                print("Orientation decoded")

                values = self.decode_green_sectors_debug(centroid_group_id)
                print("Green sectors decoded")
                
                checksum = values.pop()
                if sum(values) != checksum:
                    print(f"Checksum verification failed: {sum(values)} != {checksum}")
                else:
                    print(f"Results for centroid group {centroid_group_id}: {values}")
                    results.append(values)

            self.original_img = img.copy()
            self.plot_normal_vectors(output_dir=self.debug_dir)

            if results:
                return results
            return None

        # Non-debug mode
        self.lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        centroids = self.find_red_circles()
        
        if len(centroids) < 4:
            return None
        
        self.group_centroids(centroids)
        
        results = []
        for centroid_group_id in self.centroid_group_ids:
            self.perspective_transform(self.centroid_groups[centroid_group_id], centroid_group_id)
            self.decode_orientation()
            values = self.decode_green_sectors()
            checksum = values.pop()

            if sum(values) != checksum:
                contin
            else:
                results.append(values)
        
        if self.plot_normals:
            self.original_img = img.copy()
            self.plot_normal_vectors(output_dir=".")
        
        if results:
            return results
        return None

def main() -> None:
    parser = argparse.ArgumentParser(description='Detect PinTag from Images')
    parser.add_argument('--input_image', type=str, required=True, help='Path to the PinTag image')
    parser.add_argument('--debug', action='store_true', help='Directory to save output images')
    parser.add_argument('--plot_normals', action='store_true', help='Plot normal vectors')
    parser.add_argument('--max_ids', type=int, default=1, help='Maximum number of PinTags to detect')
    args = parser.parse_args()
    if args.debug:
        print("Running in debug mode")
        os.makedirs("debug/", exist_ok=True)
        detector = PinTagDetector(debug=True, debug_dir="debug/", max_ids=args.max_ids)
    else:
        if args.plot_normals:
            detector = PinTagDetector(plot_normals=True, max_ids=args.max_ids)
        else:
            detector = PinTagDetector(max_ids=args.max_ids)
    
    img = cv2.imread(args.input_image)
    results = detector.detect(img)

    if results:
        for result in results:
            result = f"{result[0]:02d}{result[1]:02d}{result[2]:02d}"
            print(f"Detected PinTag values: {result}")
    else:
        print("Failed to detect any PinTags")

if __name__ == "__main__":
    main()