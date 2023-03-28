import os
import shutil
import random

# Define paths for the original and new directories
orig_dir = '/home/joey/Desktop/mini_imagenet/'
train_dir = '/home/joey/Desktop/mini_imagenet_new/train'
val_dir = '/home/joey/Desktop/mini_imagenet_new/val'
meta_dir = '/home/joey/Desktop/mini_imagenet_new/meta'

# Create new directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(meta_dir, exist_ok=True)

# Define the number of images to use for training and validation per class
num_images_per_class_train = 500
num_images_per_class_val = 100

# Create lists to hold the image paths for each class
class_image_paths = {}
for class_dir in os.listdir(orig_dir):
    class_path = os.path.join(orig_dir, class_dir)
    class_image_paths[class_dir] = [os.path.join(class_path, f) for f in os.listdir(class_path)]

# Shuffle the image paths for each class
for class_dir in class_image_paths:
    random.shuffle(class_image_paths[class_dir])

# Copy the images to the train and validation directories
train_image_paths = []
val_image_paths = []
class_dict = {}
label = 0
for class_dir in class_image_paths:
    for i, image_path in enumerate(class_image_paths[class_dir]):
        if i < num_images_per_class_train:
            train_image_paths.append(image_path)
        elif i < num_images_per_class_train + num_images_per_class_val:
            val_image_paths.append(image_path)
        else:
            break
    class_dict[class_dir] = label
    label += 1


os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

for image_path in train_image_paths:
    image_name = os.path.basename(image_path)
    shutil.copy(image_path, os.path.join(train_dir, image_name))

for image_path in val_image_paths:
    image_name = os.path.basename(image_path)
    shutil.copy(image_path, os.path.join(val_dir, image_name))

# Write the image paths and labels to the meta directory
with open(os.path.join(meta_dir, 'train.txt'), 'w') as train_file:
    for image_path in train_image_paths:
        class_dir = os.path.basename(os.path.dirname(image_path))
        train_file.write(f"{image_path} {class_dict[class_dir]}\n")
with open(os.path.join(meta_dir, 'val.txt'), 'w') as val_file:
    for image_path in val_image_paths:
        class_dir = os.path.basename(os.path.dirname(image_path))
        val_file.write(f"{image_path} {class_dict[class_dir] }\n")