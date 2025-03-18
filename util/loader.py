import os
import numpy as np

import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import multiprocessing as mp


class ImageFolderWithFilename(datasets.ImageFolder):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, filename).
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        filename = path.split(os.path.sep)[-2:]
        filename = os.path.join(*filename)
        return sample, target, filename

def is_valid_file(filepath):
    """ Custom function to check file validity, implemented according to actual needs. """
    return filepath.endswith(".npz")

def process_directory(directory, class_to_idx):
    """
    Process a single directory and return the paths of valid files and their corresponding labels.
    """
    samples = []
    for root_dir, _, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(root_dir, filename)
            if is_valid_file(filepath):
                label = class_to_idx[os.path.basename(root_dir)]
                samples.append((filepath, label))
    return samples

def list_valid_files(root, is_valid_file):
    """
    Use multiprocessing to get valid file paths and build a (path, label) list.
    """
    all_samples = []
    
    # Get sub-folder list, the label is the index of the sub-folder
    classes, class_to_idx = datasets.folder.find_classes(root)
    
    # Use a multi-process pool to process each sub-folder
    with mp.Pool(processes=48) as pool:
        subdirs = [os.path.join(root, d) for d in classes]
        results = pool.starmap(process_directory, [(subdir, class_to_idx) for subdir in subdirs])
        
        # Aggregate the results of all sub-processes
        for res in results:
            all_samples.extend(res)
    
    return all_samples

class CachedFolder(datasets.DatasetFolder):
    def __init__(self, root: str):
        # Use multi-process to load all valid files and labels
        self.samples = list_valid_files(root, is_valid_file)
        
        # Save subclass information
        self.root = root
        self.loader = None  # npz data format may not require a custom loader
        self.extensions = None

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (moments, target)
        """
        path, target = self.samples[index]
        data = np.load(path)
        
        # Random horizontal flip
        if torch.rand(1) < 0.5:
            moments = data['moments']
        else:
            moments = data['moments_flip']
        
        return moments, target

    def __len__(self):
        return len(self.samples)