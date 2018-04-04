"""Dataset setting and data loader for VisDA.

Modified from the class:
    - torchvision.datasets.ImageFolder
"""

import os
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import params

class VisDA(data.Dataset):
    """VisDA Dataset.
    Args:
        root (string): Root directory of dataset where dataset file exist.
        sub_dir (string): one of the {train | validation | test}.
        split (string, optional): one of the {all | train | test}.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``

    Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        images (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, sub_dir, split='all', transform=None):
        """Init VisDA dataset."""
        # Num of Train = 152397, validation = 55387, test = 72372
        self.root = os.path.expanduser(root)
        self.transform = transform
        if split == "all":
            filename = 'image_list.txt'
        elif split == 'train':
            filename = 'image_list_train.txt'
        elif split == "test":
            filename = 'image_list_veri.txt'
        self.filename = os.path.join(self.root, sub_dir, filename)
        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You should download the dataset and move it to <dataroot>")
        images = []
        with open(self.filename, 'r') as f:
            for line in f:
                path, class_idx = line.strip().split()
                abspath = os.path.join(self.root, sub_dir, path)
                images.append((abspath, class_idx))

        self.images = images
        self.dataset_size = len(images)
        classes, class_to_idx = self.find_classes(os.path.join(self.root, 'train'))
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __getitem__(self, index):
        """Get images and target for data loader.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        path, target = self.images[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, int(target)

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(self.filename)

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

def get_visda(root, sub_dir, split='all'):
    """Get VisDA dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize([params.image_size, params.image_size]),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=params.dataset_mean,
                                          std=params.dataset_std)])

    # dataset and data loader
    visda_dataset = VisDA(root=root,
                          sub_dir=sub_dir,
                          split=split,
                          transform=pre_process)

    visda_data_loader = torch.utils.data.DataLoader(
        dataset=visda_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        pin_memory=True)

    return visda_data_loader

if __name__ == '__main__':
    # tmp = VisDA('/media/Data/dataset_xian/VisDA')
    # data, label = tmp[1]
    # data.show(data)
    # print(label)
    src_data_loader = get_visda(root=params.data_root, sub_dir='train', split='train')
    src_data_loader_eval = get_visda(root=params.data_root, sub_dir='train', split='test')

    count = 0
    for data, lable in src_data_loader:
        count += data.size()[0]
        print('count:{}'.format(count))
        if count > 10000:
            break

    count_test = 0
    for data, lable in src_data_loader_eval:
        count_test += data.size()[0]
        print('count_test:{}'.format(count_test))