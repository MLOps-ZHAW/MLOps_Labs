import os
import argparse
import ast
import albumentations as A
from PIL import Image
import numpy as np


def parse_value(value):
    """Turn "200" into 200, "0.5" into 0.5, "True" into True; keep anything else as a string."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def apply_augmentation(image_path, save_dir, augmentation, suffix):
    image = np.array(Image.open(image_path).convert("RGB"))
    augmented = augmentation(image=image)["image"]
    # Most augmentations have a probability `p` - skip the images the augmentation was not applied to.
    # (Re-saving an unchanged JPEG would still change the file, because JPEG compression is lossy.)
    if augmented.shape == image.shape and np.array_equal(augmented, image):
        return False
    # Save the augmented image as a new file next to the original one, e.g. image_00001_HorizontalFlip.jpg
    stem, ext = os.path.splitext(os.path.basename(image_path))
    Image.fromarray(augmented).save(os.path.join(save_dir, f"{stem}_{suffix}{ext}"))
    return True


def main(args):
    augmentation = getattr(A, args.augmentation)(**args.augmentation_params)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # List the files first, so that images written during this run are not augmented again
    filenames = sorted(f for f in os.listdir(args.input_dir) if f.endswith(('.jpg', '.png')))
    n_augmented = 0
    for filename in filenames:
        image_path = os.path.join(args.input_dir, filename)
        n_augmented += apply_augmentation(image_path, args.save_dir, augmentation, args.augmentation)
    print(f"Augmented {n_augmented} of {len(filenames)} images, saved to {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Albumentations augmentations to a directory of images.")
    parser.add_argument("input_dir", type=str, help="Path to the directory containing input images.")
    parser.add_argument("save_dir", type=str, help="Path to the directory to save augmented images (can be input_dir).")
    parser.add_argument("--augmentation", type=str, default="HorizontalFlip",
                        choices=[name for name in dir(A) if name[0].isupper()],
                        help="Name of the augmentation class.")
    parser.add_argument("--augmentation_params", type=str, nargs='*', default=[],
                        help="Parameters for the augmentation in the format key1=value1 key2=value2 ...")

    args = parser.parse_args()

    args.augmentation_params = dict(item.split('=', 1) for item in args.augmentation_params)
    args.augmentation_params = {key: parse_value(value) for key, value in args.augmentation_params.items()}

    main(args)
