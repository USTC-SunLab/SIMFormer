from utils_imageJ import save_tiff_imagej_compatible
from skimage.io import imread
from skimage.transform import rescale
import mrc
import os
import glob
import numpy as np
import tqdm

# Set BioSR dataset path
BIOSR_PATH = "/root/data/BioSR"

for TASK in ["CCPs", "Microtubules", "F-actin"]:
    if TASK == "CCPs":
        names = ["level_%02d" % i for i in range(1, 10)]
    elif TASK == "Microtubules":
        names = ["level_%02d" % i for i in range(1, 10)]
    elif TASK == "F-actin":
        names = ["level_%02d" % i for i in range(1, 13)]
    else:
        raise ValueError("Invalid TASK")

    for name in names:
        # Process 2D
        file_dir = os.path.join(BIOSR_PATH, "%s/Raw"%TASK)
        save_path = os.path.join(BIOSR_PATH, "%s/2D/%s"%(TASK, name))

        os.makedirs(save_path, exist_ok=True)

        file_paths = glob.glob(os.path.join(file_dir, "*", "RawSIMData_%s.mrc"%name))
        IMG = []

        for file_path in file_paths:
            img = np.float32(mrc.imread(file_path))
            print(file_path, img.shape)
            save_tiff_imagej_compatible(os.path.join(save_path, file_path.split(os.sep)[-2] + ".tif"), img.astype(np.float32), "TYX")

        # Process widefield
        file_dir = os.path.join(BIOSR_PATH, "%s/Raw"%TASK)
        save_path = os.path.join(BIOSR_PATH, "%s/WF/%s"%(TASK, name))

        os.makedirs(save_path, exist_ok=True)

        file_paths = glob.glob(os.path.join(file_dir, "*", "RawSIMData_%s.mrc"%name))
        IMG = []

        for file_path in file_paths:
            img = np.float32(mrc.imread(file_path))
            img = img.mean(axis=0, keepdims=False)
            print(file_path, img.shape)
            save_tiff_imagej_compatible(os.path.join(save_path, file_path.split(os.sep)[-2] + ".tif"), img.astype(np.float32), "YX")

# ER 2D
names = ["level_%02d" % i for i in range(1, 7)]
for name in names:
    file_dir = os.path.join(BIOSR_PATH, "ER/Raw")
    save_path = os.path.join(BIOSR_PATH, "ER/2D/%s"%name)

    os.makedirs(save_path, exist_ok=True)
    
    file_paths = glob.glob(os.path.join(file_dir, "*", "RawSIMData", "RawSIMData_%s.mrc"%name))

    for file_path in file_paths:
        img = np.float32(mrc.imread(file_path))
        img = img[:, ::-1, :]
        print(file_path, img.shape)
        save_tiff_imagej_compatible(os.path.join(save_path, file_path.split(os.sep)[-3] + ".tif"), img.astype(np.float32), "TYX")

    # ER widefield
    file_dir = os.path.join(BIOSR_PATH, "ER/Raw")
    save_path = os.path.join(BIOSR_PATH, "ER/WF/%s"%name)

    os.makedirs(save_path, exist_ok=True)
    file_paths = glob.glob(os.path.join(file_dir, "*", "RawSIMData", "RawSIMData_%s.mrc"%name))

    for file_path in file_paths:
        img = np.float32(mrc.imread(file_path))
        img = img[:, ::-1, :]
        img = img.mean(axis=0, keepdims=False)
        print(file_path, img.shape)
        save_tiff_imagej_compatible(os.path.join(save_path, file_path.split(os.sep)[-3] + ".tif"), img.astype(np.float32), "YX")

# Generate train/test splits for highest noise level
print("\nGenerating train/test splits for highest noise level...")

# Define highest levels for each task
highest_levels = {
    "CCPs": "level_09",
    "Microtubules": "level_09", 
    "F-actin": "level_12",
    "ER": "level_06"
}

# Process each task
for TASK, highest_level in highest_levels.items():
    print(f"\nProcessing {TASK} - {highest_level}")
    
    # Create train/test directories
    train_path = os.path.join(BIOSR_PATH, f"{TASK}/2D/train")
    test_path = os.path.join(BIOSR_PATH, f"{TASK}/2D/test")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    # Process cells
    if TASK == "ER":
        # ER has different structure
        file_paths = glob.glob(os.path.join(BIOSR_PATH, f"{TASK}/Raw/*/RawSIMData/RawSIMData_{highest_level}.mrc"))
    else:
        file_paths = glob.glob(os.path.join(BIOSR_PATH, f"{TASK}/Raw/*/RawSIMData_{highest_level}.mrc"))
    
    for file_path in sorted(file_paths):
        # Extract cell name
        if TASK == "ER":
            cell_name = file_path.split(os.sep)[-3]
        else:
            cell_name = file_path.split(os.sep)[-2]
        
        # Extract cell number
        cell_num = int(cell_name.split("_")[1])
        
        # Determine train or test
        if cell_num <= 35:
            save_dir = train_path
        elif cell_num <= 54:
            save_dir = test_path
        else:
            continue  # Skip cells beyond 54
        
        # Read and save
        img = np.float32(mrc.imread(file_path))
        if TASK == "ER":
            img = img[:, ::-1, :]  # Flip for ER
        
        save_file = os.path.join(save_dir, f"{cell_name}.tif")
        save_tiff_imagej_compatible(save_file, img.astype(np.float32), "TYX")
        print(f"  {cell_name} -> {'train' if cell_num <= 35 else 'test'}")

print("\nDone!")
