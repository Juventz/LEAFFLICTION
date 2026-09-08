#!/usr/bin/env python3

import argparse
import os
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from typing import Dict

from image_utils import is_image_file


def analyze_directory(root: str) -> Dict[str, Counter]:
    data = defaultdict(Counter)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"The specified path does not exist: {root}")
    entries = sorted(os.listdir(root))
    has_underscore_format = any(
        '_' in name
        for name in entries
        if os.path.isdir(os.path.join(root, name))
    )
    if has_underscore_format:
        for entry in entries:
            subpath = os.path.join(root, entry)
            if not os.path.isdir(subpath):
                continue
            parts = entry.split('_')
            plant = parts[0]
            state = '_'.join(parts[1:]) if len(parts) > 1 else 'unknown'
            count = 0
            try:
                for fname in os.listdir(subpath):
                    fpath = os.path.join(subpath, fname)
                    if os.path.isfile(fpath) and is_image_file(fname):
                        count += 1
            except PermissionError:
                continue
            if count > 0:
                data[plant][state] += count
        return data
    plant = os.path.basename(os.path.normpath(root))
    for entry in entries:
        subpath = os.path.join(root, entry)
        if not os.path.isdir(subpath):
            continue
        state = entry
        count = 0
        try:
            for fname in os.listdir(subpath):
                fpath = os.path.join(subpath, fname)
                if os.path.isfile(fpath) and is_image_file(fname):
                    count += 1
        except PermissionError:
            continue
        if count > 0:
            data[plant][state] += count
    return data


def plot_distribution(plant: str, counter: Counter):
    labels = list(counter.keys())
    counts = [counter[label] for label in labels]
    total = sum(counts)
    if total == 0:
        raise ValueError(f"No images found for {plant}")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(len(labels))]
    axes[0].pie(counts, labels=labels, autopct='%1.1f%%',
                startangle=90, colors=colors)
    axes[0].axis('equal')
    axes[0].set_title(f'{plant.capitalize()} class distribution')
    axes[1].bar(range(len(labels)), counts, color=colors)
    axes[1].set_title(f"{plant.capitalize()} class distribution")
    axes[1].set_ylabel("Image count")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def find_plant_in_images(
    images_root: str,
    plant_name: str,
) -> Dict[str, Counter]:
    data = defaultdict(Counter)
    target = plant_name.lower()
    for entry in sorted(os.listdir(images_root)):
        subpath = os.path.join(images_root, entry)
        if not os.path.isdir(subpath):
            continue
        if entry.lower().startswith(target + '_'):
            parts = entry.split('_')
            plant = parts[0]
            state = '_'.join(parts[1:]) if len(parts) > 1 else 'unknown'
            count = 0
            try:
                for fname in os.listdir(subpath):
                    fpath = os.path.join(subpath, fname)
                    if os.path.isfile(fpath) and is_image_file(fname):
                        count += 1
            except PermissionError:
                continue
            if count > 0:
                data[plant][state] += count
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Analyze the image distribution by plant and state.")
    parser.add_argument(
        "root",
        help="Path to the directory to analyze or plant name (example: Apple)",
    )
    args = parser.parse_args()
    
    root = args.root
    images_dir = os.path.join(os.getcwd(), 'images')
    
    if os.path.isdir(root):
        data = analyze_directory(root)
    elif os.path.isdir(os.path.join(os.getcwd(), root)):
        data = analyze_directory(os.path.join(os.getcwd(), root))
    elif os.path.isdir(images_dir):
        data = find_plant_in_images(images_dir, root)
        if not data:
            print(
                f"No plant directory named '{root}' found in '{images_dir}'."
            )
            return
    else:
        print(
            "The specified path does not exist and the 'images' directory "
            f"was not found: {root}"
        )
        return
    if not data:
        print(
            "No data found. Check the path and its subdirectories."
        )
        return
    for plant, counter in data.items():
        try:
            plot_distribution(plant, counter)
        except Exception as e:
            print(f"Error for {plant}: {e}")


if __name__ == '__main__':
    main()
