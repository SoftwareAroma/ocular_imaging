import os
import argparse
import kaggle
import keras
import numpy as np


def download_kaggle_dataset(dataset_name, dataset_path) -> None:
    """
        Download kaggle dataset
        :param dataset_name: name of the dataset
        :param dataset_path: path to download the dataset
        :return: None
    """
    # if the dataset_path contains the dataset_name, alert the user and skip the download
    if dataset_name in dataset_path:
        print("Dataset already downloaded")
        return
    else:
        print("Downloading dataset...")
        kaggle.api.authenticate()
        # create the folder for the dataset_path if it does not exist
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        # Download the dataset
        kaggle.api.dataset_download_files(dataset_name, path=dataset_path, unzip=True)
        print("Dataset downloaded successfully.")
    
    # Define the target image size
    image_size = (128, 128)
    
    images = []
    
    # Walk through the directory structure
    for subdir, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                img_path = os.path.join(subdir, file)
                img = keras.preprocessing.image.load_img(img_path, target_size=image_size)
                img = keras.preprocessing.image.img_to_array(img)
                img = img / 255.0  # Normalize to [0, 1]
                images.append(img)

    # Convert list to numpy array
    images = np.array(images)
    print("Number of images loaded:", len(images))
    

def main():
    # parse the arguments to take the dataset name and path
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset')
    parser.add_argument('--dataset_path', type=str, help='Path to download the dataset')
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    # Download the dataset
    download_kaggle_dataset(dataset_name=dataset_name, dataset_path=dataset_path)
    
    # sample usage
    # python get_dataset.py --dataset_name 'linchundan/fundusimage1000' --dataset_path 'datasets/fundusimage1000'
    
if __name__ == '__main__':
    main()
    