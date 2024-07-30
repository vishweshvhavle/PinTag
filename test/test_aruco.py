import os
import sys
import cv2
import apriltag
import cv2.aruco as aruco
import json

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define the ArUco detection function
def detect_aruco(img):
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(img)
    print(ids)
    return [{'id': id[0]} for id in ids] if ids is not None else None

def main():
    dataset_path = 'data/aruco_tags'
    image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if image_files:
        img_file = image_files[0]
        img_path = os.path.join(dataset_path, img_file)
        img = cv2.imread(img_path)

        try:
            result = detect_aruco(img)
            if result:
                print(f"Detection result for {img_file}: {result}")
            else:
                print(f"No tag detected in {img_file}")
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
    else:
        print("No image files found in the dataset.")

if __name__ == "__main__":
    main()
