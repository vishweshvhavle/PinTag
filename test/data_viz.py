import os
import sys
import cv2
import matplotlib.pyplot as plt

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_image(img_path):
    if os.path.exists(img_path):
        return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    else:
        print(f"Image not found: {img_path}")
        return None

def visualize_datasets():
    datasets = ['pin_tags_random', 'april_tags_random', 'aruco_tags_random']
    base_path = "data"

    # Create a figure with 3 rows and 3 columns per dataset
    num_datasets = len(datasets)
    fig, axes = plt.subplots(num_datasets, 3, figsize=(15, 5 * num_datasets))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    for i, dataset in enumerate(datasets):
        dataset_path = os.path.join(base_path, dataset)
        frame_numbers = [f for f in os.listdir(dataset_path) if f.endswith('.png')][:3]  # Take first 3 images

        for j, frame_number in enumerate(frame_numbers):
            img_path = os.path.join(dataset_path, frame_number)
            img = load_image(img_path)
            if img is not None:
                axes[i, j].imshow(img)
                axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig('tag_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    visualize_datasets()
