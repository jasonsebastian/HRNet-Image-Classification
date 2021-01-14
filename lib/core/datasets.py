import os
import re

from PIL import Image
from torch.utils.data import Dataset


class ImagePersonDataset(Dataset):
    """Image Person ReID dataset."""

    def __init__(self, config, transform=None):
        self.config = config
        self.images_person = self.get_images_person()
        self.transform = transform

    def __len__(self):
        return len(self.images_person)

    def __getitem__(self, idx):
        img_name, label = self.images_person[idx]
        original_img_path = os.path.join(self.config.DATASET.ORIGINAL_ROOT,
                                         self.config.DATASET.TRAIN_SET,
                                         img_name)
        downsampled_img_path = os.path.join(self.config.DATASET.DOWNSAMPLED_ROOT,
                                            self.config.DATASET.TRAIN_SET,
                                            img_name)
        original_image = Image.open(original_img_path).convert('RGB')
        downsampled_image = Image.open(downsampled_img_path).convert('RGB')
        
        if self.transform:
            original_image, downsampled_image = self.transform(original_image, downsampled_image)
        
        return {'original_image': original_image,
                'downsampled_image': downsampled_image,
                'label': label}

    def get_images_person(self):
        img_filenames = [x for x in os.listdir(os.path.join(self.config.DATASET.DOWNSAMPLED_ROOT,
                                                            self.config.DATASET.TRAIN_SET))
                         if x.endswith('.{}'.format(self.config.DATASET.DATA_FORMAT))]
        pattern = re.compile(r'([-\d]+)_c')

        pid_container = set()
        for img_name in img_filenames:
            pid, = map(int, pattern.search(img_name).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        images_person = []
        for img_name in img_filenames:
            pid, = map(int, pattern.search(img_name).groups())
            if pid == -1: continue  # junk images are just ignored
            label = pid2label[pid]
            images_person.append((img_name, label))

        return images_person
        
def show_image(image):
    plt.imshow(image)
