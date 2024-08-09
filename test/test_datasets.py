import os
import sys
import cv2
from tqdm import tqdm
import apriltag
import cv2.aruco as aruco
import json
import time
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import the detector
from detector.detector import PinTagDetector

def profile_detection(detector_func, img):
    detector_func(img)
    start_time = time.time()
    result = detector_func(img)
    end_time = time.time()
    detection_time = end_time - start_time
    return result, detection_time

def process_dataset(dataset_path, detector_func, log_file, output_log_file):
    successful_detections = 0
    failed_frames = []
    outputs = {}
    detection_times = []

    image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in tqdm(image_files, desc=f"Processing {os.path.basename(dataset_path)}"):
        img_path = os.path.join(dataset_path, img_file)
        img = cv2.imread(img_path)

        try:
            result, detection_time = profile_detection(detector_func, img)
            detection_times.append(detection_time)
            if result:
                successful_detections += 1
                outputs[img_file] = result
            else:
                failed_frames.append(f"{img_file}: No tag detected")
        except Exception as e:
            failed_frames.append(f"{img_file}: {str(e)}")

    with open(log_file, 'w') as f:
        f.write("\n".join(failed_frames))

    with open(output_log_file, 'w') as f:
        json.dump(outputs, f, indent=2)

    return successful_detections, len(failed_frames), detection_times

def detect_pintag(img):
    detector = PinTagDetector()
    result = detector.detect(img)
    return result if result else None

def detect_apriltag(img):
    detector = apriltag.Detector(apriltag.DetectorOptions(families='tag36h10'))
    result = detector.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return [{'id': tag.tag_id} for tag in result] if result else None

def detect_aruco(img):
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(img)
    return [{'id': int(id[0])} for id in ids] if ids is not None else None

def get_off_axis_angle(filename):
    parts = filename[:-4].split('_')
    if len(parts) < 5:
        return 0.0
    
    theta = float(parts[3])
    phi = float(parts[5])
    return 90 - phi

def get_target_distance(filename):
    parts = filename[:-4].split('_')
    if len(parts) < 5:
        return 0.0
    
    radius = float(parts[7])
    return radius

def analyze_experiment1(datasets):
    plt.figure(figsize=(12, 8))

    for dataset_name, (dataset_path, detector_func, name, color) in datasets.items():
        results = {i: [] for i in range(90)}

        for filename in tqdm(os.listdir(dataset_path), desc=f"Processing {name} Experiment 1"):
            off_axis_angle = get_off_axis_angle(filename)

            try:
                result = detector_func(cv2.imread(os.path.join(dataset_path, filename)))
                if result:
                    results[int(off_axis_angle)].append(1)
                else:
                    results[int(off_axis_angle)].append(0)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

        angles = list(results.keys())
        accuracies = [sum(values) / len(values) if values else 0 for values in results.values()]

        plt.plot(angles, accuracies, color=color, label=name)

    plt.xlabel("Off-axis Angle (degrees)")
    plt.ylabel("Detection Accuracy")
    plt.title("Experiment 1: Varying Off-axis Angle")
    plt.legend()
    plt.savefig("experiment1_combined.png")

def analyze_experiment2(datasets):
    plt.figure(figsize=(12, 8))

    for dataset_name, (dataset_path, detector_func, name, color) in datasets.items():
        results = {i: [] for i in range(1, 11)}

        for filename in tqdm(os.listdir(dataset_path), desc=f"Processing {name} Experiment 2"):
            target_distance = get_target_distance(filename)

            try:
                result = detector_func(cv2.imread(os.path.join(dataset_path, filename)))
                if result:
                    results[int(target_distance)].append(1)
                else:
                    results[int(target_distance)].append(0)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

        distances = list(results.keys())
        accuracies = [sum(values) / len(values) if values else 0 for values in results.values()]

        plt.plot(distances, accuracies, color=color, label=name)

    plt.xlabel("Target Distance (meters)")
    plt.ylabel("Detection Accuracy")
    plt.title("Experiment 2: Varying Target Distance")
    plt.legend()
    plt.savefig("experiment2_combined.png")

def main():
    datasets = {
        'pin_tags': ('data/pin_tags_random', detect_pintag, 'Pin Tags', (1, 0, 0)),
        'april_tags': ('data/april_tags_random', detect_apriltag, 'April Tags', (0, 1, 0)),
        'aruco_tags': ('data/aruco_tags_random', detect_aruco, 'Aruco Tags', (0, 0, 1))
    }

    total_results = {}

    for dataset_name, (dataset_path, detector_func, name, color) in datasets.items():
        successful, failed, detection_times = process_dataset(dataset_path, detector_func, f"failed_frames_logs_{dataset_name}.txt", f"output_logs_{dataset_name}.json")
        total_results[dataset_name] = {
            'successful': successful,
            'failed': failed,
            'total': successful + failed,
            'detection_times': detection_times
        }

    # analyze_experiment1(datasets)
    # analyze_experiment2(datasets)

    print("\nDetection Results:")
    for dataset_name, results in total_results.items():
        detection_times = results['detection_times']
        if detection_times:
            min_time = min(detection_times)
            avg_time = sum(detection_times) / len(detection_times)
            max_time = max(detection_times)
        else:
            min_time = avg_time = max_time = 0
        print(f"{datasets[dataset_name][2]}:")
        print(f"  Successful detections: {results['successful']}")
        print(f"  Failed detections: {results['failed']}")
        print(f"  Total images: {results['total']}")
        print(f"  Success rate: {results['successful'] / results['total'] * 100:.2f}%")
        print(f"  Min detection time: {min_time:.4f}s")
        print(f"  Avg detection time: {avg_time:.4f}s")
        print(f"  Max detection time: {max_time:.4f}s")
        print()

if __name__ == "__main__":
    main()
