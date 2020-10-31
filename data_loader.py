import torch
import pandas as pd
from skimage import io
import os
import numpy as np


class Dataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, root_dir):
        """
        Initialize the config file with images and the root directory for tha data.
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        self.frames = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        """
        Return the length of tha dataset.
        :return: number of samples available.
        """
        return len(self.frames)

    def __getitem__(self, idx):
        """
        Get an image and label. Image is prepared to color channel first and then width and height.
        :param idx: index of the item.
        :return:
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.frames.iloc[idx, 1])
        image = np.array(io.imread(img_name))
        image = np.rollaxis(image, 2, 0)  # need to change to color channel first, as this is the way the network
        # wants it
        labels = self.frames.iloc[idx, 2]
        labels = np.array(labels)
        sample = {'image': image, 'labels': labels}

        return sample
