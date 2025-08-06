import os
import shutil

# Source directories
figshare_dir = '/Users/priyanshuparashar/braintumor/datasets/figshare_converted'
kaggle_dir = '/Users/priyanshuparashar/braintumor/datasets/Kaggle_Brain_MRI_Images'

# Output combined directory
combined_dir = '/Users/priyanshuparashar/braintumor/datasets/combined_dataset'
tumor_dir = os.path.join(combined_dir, 'tumor')
no_tumor_dir = os.path.join(combined_dir, 'no_tumor')

# Make output folders
os.makedirs(tumor_dir, exist_ok=True)
os.makedirs(no_tumor_dir, exist_ok=True)

def copy_images(source_folder, target_folder, prefix):
    count = 0
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            src = os.path.join(source_folder, filename)
            dst = os.path.join(target_folder, f"{prefix}_{count}.jpg")
            shutil.copyfile(src, dst)
            count += 1
    print(f"✅ Copied {count} images from {source_folder} → {target_folder}")

# Copy Figshare images
copy_images(os.path.join(figshare_dir, 'benign'), tumor_dir, 'fig_benign')
copy_images(os.path.join(figshare_dir, 'malignant'), tumor_dir, 'fig_malignant')
copy_images(os.path.join(figshare_dir, 'no_tumor'), no_tumor_dir, 'fig_no')

# Copy Kaggle images
copy_images(os.path.join(kaggle_dir, 'yes'), tumor_dir, 'kaggle_yes')
copy_images(os.path.join(kaggle_dir, 'no'), no_tumor_dir, 'kaggle_no')
