import os
import shutil
import math

def reduce_dataset_size(folder_path, reduction_factor=5):
    """
    Reduces the number of images in each category to 1/5 of the original count.
    
    Args:
        folder_path: Path to the cat2dog_v2 folder
        reduction_factor: Factor by which to reduce the dataset size
    """
    categories = ['cat', 'dog']
    splits = ['train', 'test']
    
    for category in categories:
        for split in splits:
            # Path to the specific category and split folder
            target_folder = os.path.join(folder_path, split, category)
            
            if not os.path.exists(target_folder):
                print(f"Folder {target_folder} does not exist, skipping...")
                continue
                
            # Get all files in the folder
            files = [f for f in os.listdir(target_folder) if os.path.isfile(os.path.join(target_folder, f))]
            
            # Calculate how many files to keep
            files_to_keep = math.ceil(len(files) / reduction_factor)
            files_to_remove = files[files_to_keep:]
            
            print(f"Reducing {target_folder} from {len(files)} to {files_to_keep} files")
            
            # Remove the excess files
            for file in files_to_remove:
                os.remove(os.path.join(target_folder, file))

if __name__ == "__main__":
    cat2dog_folder = "/Users/raycheng/Desktop/Project/dataset/cat2dog_v2"
    reduce_dataset_size(cat2dog_folder)
    print("Dataset reduction complete")
