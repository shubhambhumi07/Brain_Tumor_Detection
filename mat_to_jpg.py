import os
import h5py
import numpy as np
import cv2

source_dir = '/Users/priyanshuparashar/braintumor/datasets/Figshare_Brain_MRI_Dataset'
output_dir = '/Users/priyanshuparashar/braintumor/datasets/figshare_converted'

label_map = {1: 'no_tumor', 2: 'benign', 3: 'malignant'}

# Create output folders
for label_name in label_map.values():
    os.makedirs(os.path.join(output_dir, label_name), exist_ok=True)

converted_count = 0

# Loop through .mat files
for filename in os.listdir(source_dir):
    if filename.endswith('.mat'):
        file_path = os.path.join(source_dir, filename)
        print(f"📂 Processing file: {filename}")

        try:
            with h5py.File(file_path, 'r') as f:
                cjdata = f['cjdata']
                image = np.array(cjdata['image'])
                label = int(np.array(cjdata['label'])[0][0])
        except Exception as e:
            print(f"⚠️ Skipping {filename}: {e}")
            continue

        # Normalize and convert to uint8
        image = image.astype(np.float32)
        image -= image.min()
        image /= image.max()
        image *= 255.0
        image = image.astype(np.uint8)

        # Convert to 3 channels
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Save image
        label_name = label_map.get(label, 'unknown')
        output_path = os.path.join(output_dir, label_name, filename.replace('.mat', '.jpg'))
        cv2.imwrite(output_path, image)
        converted_count += 1

print(f"\n✅ Done! Successfully converted {converted_count} images.")
