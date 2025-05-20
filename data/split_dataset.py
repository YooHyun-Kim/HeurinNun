import os
import shutil
import random
from pathlib import Path

def split_dataset(root_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    test_ratio = 1.0 - train_ratio - val_ratio
    class_names = os.listdir(root_dir)
    for class_name in class_names:
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = []
        for ext in ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']:
            images += list(Path(class_path).glob(f"*.{ext}"))

        random.shuffle(images)
        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        for phase, image_list in zip(['train', 'val', 'test'], [train_images, val_images, test_images]):
            target_dir = os.path.join(output_dir, phase, class_name)
            os.makedirs(target_dir, exist_ok=True)
            for img_path in image_list:
                shutil.copy(img_path, os.path.join(target_dir, img_path.name))

    print("\n✅ Dataset split 완료!")
    print(f"Train: {train_ratio*100:.0f}%, Val: {val_ratio*100:.0f}%, Test: {test_ratio*100:.0f}%")
    print(f"출력 경로: {output_dir}/train, /val, /test")


if __name__ == "__main__":
    original_dir = "prototypes"
    target_dir = "image_dataset"
    split_dataset(original_dir, target_dir, train_ratio=0.8, val_ratio=0.1)