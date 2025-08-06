import os
import shutil
import random

source_dir = '/Users/priyanshuparashar/braintumor/datasets/combined_dataset'
output_base = '/Users/priyanshuparashar/braintumor/datasets/combined_dataset_split'

categories = ['tumor', 'no_tumor']
split_ratio = (0.7, 0.15, 0.15)  # train, val, test

for category in categories:
    src_folder = os.path.join(source_dir, category)
    images = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)
    
    n_total = len(images)
    n_train = int(split_ratio[0] * n_total)
    n_val = int(split_ratio[1] * n_total)

    splits = {
        'train': images[:n_train],
        'val': images[n_train:n_train + n_val],
        'test': images[n_train + n_val:]
    }

    for split_name, split_files in splits.items():
        out_dir = os.path.join(output_base, split_name, category)
        os.makedirs(out_dir, exist_ok=True)
        for file in split_files:
            shutil.copyfile(
                os.path.join(src_folder, file),
                os.path.join(out_dir, file)
            )
        print(f"✅ {split_name}/{category}: {len(split_files)} images")
