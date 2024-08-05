import os
import sys
import cv2
from tqdm import tqdm
import apriltag
import cv2.aruco as aruco
import json
import time

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import the detector
from detector.detector import PinTagDetector

def profile_detection(detector_func, img):
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
    detector = apriltag.Detector(apriltag.DetectorOptions(families='tag16h5'))
    result = detector.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return [{'id': tag.tag_id} for tag in result] if result else None

def detect_aruco(img):
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(img)
    return [{'id': int(id[0])} for id in ids] if ids is not None else None

def main():
    datasets = {
        'india_tags': ('data/pin_tags', detect_pintag, 'failed_frames_logs_pintag.txt', 'output_logs_pintag.json'),
        'april_tags': ('data/april_tags', detect_apriltag, 'failed_frames_logs_apriltag.txt', 'output_logs_apriltag.json'),
        'aruco_tags': ('data/aruco_tags', detect_aruco, 'failed_frames_logs_aruco.txt', 'output_logs_aruco.json')
    }

    total_results = {}

    for dataset_name, (dataset_path, detector_func, log_file, output_log_file) in datasets.items():
        successful, failed, detection_times = process_dataset(dataset_path, detector_func, log_file, output_log_file)
        total_results[dataset_name] = {
            'successful': successful,
            'failed': failed,
            'total': successful + failed,
            'detection_times': detection_times
        }

    print("\nDetection Results:")
    for dataset_name, results in total_results.items():
        detection_times = results['detection_times']
        if detection_times:
            min_time = min(detection_times)
            avg_time = sum(detection_times) / len(detection_times)
            max_time = max(detection_times)
        else:
            min_time = avg_time = max_time = 0
        print(f"{dataset_name.capitalize()}:")
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
