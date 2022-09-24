"""
CREATE DATASETS
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915
from xml.dom import minidom
import torch.utils.data as data
import torch
from random import shuffle
from torchvision import transforms
from torchvision.datasets import DatasetFolder

from pathlib import Path
from PIL import Image
import numpy as np
import os
import os.path
import random
import imageio
import numpy as np

# pylint: disable=E1101

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, nz=100, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.noise = torch.FloatTensor(len(self.imgs), nz, 1, 1).normal_(0, 1)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        latentz = self.noise[index]

        # TODO: Return these variables in a dict.
        # return img, latentz, index, target
        return {'image': img, 'latentz': latentz, 'index': index, 'frame_gt': target}

    def __setitem__(self, index, value):
        self.noise[index] = value

    def __len__(self):
        return len(self.imgs)
    
    


def check_paths(*args, names=None):
    """checks if the given paths exist.

    Args:
        args (list of paths): the paths to be checked

    Raises:
        ValueError: thrown if one of the given paths does not exist
    """
    for idx, path in enumerate(args):
        path = Path(path)
        if not path.exists():
            if names is not None and idx < len(names):
                raise ValueError(f"{str(names[idx])}: {str(path)} does not exist!")
            else:
                raise ValueError(f"file {str(path)} does not exist!")


class MVTecDataset:

    def __init__(self, data_dir, nz=100, inference=False, train=True, include_random_images=False):
        self.data_dir = Path(data_dir)
        check_paths(self.data_dir)
        if not self.data_dir.is_dir():
            raise RuntimeError(f"direcotry {str(self.data_dir.resolve())} is not a directory!")
        
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256, 256)),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.train = train
        
        self.inference = inference
        if self.train:
            self.image_files = list((self.data_dir / "train" / "good").iterdir())
        else:
            self.image_files = []
            for directory in (self.data_dir / "test").iterdir():
                self.image_files += list(directory.iterdir())
            if include_random_images:
                other_cats = list(self.data_dir.parent.iterdir())
                random_files = []
                selected = other_cats[random.randint(0, len(other_cats) - 1)]
                print("got selected:", str(selected), other_cats, random.randint(0, len(other_cats) - 1))
                for directory in (selected / "test").iterdir():
                    if directory.name != "good":
                        random_files += list(directory.iterdir())
                for _ in range(40):
                    self.image_files.append(random_files[random.randint(0, len(random_files) - 1)])
                train_image_files = list((self.data_dir / "train" / "good").iterdir())
                for _ in range(40):
                    self.image_files.append(train_image_files[random.randint(0, len(train_image_files) - 1)])
                
        self.image_files = sorted(self.image_files)
        self.image_files = [image for image in self.image_files if image.suffix == ".png"]
        self.noise = torch.FloatTensor(len(self.image_files), nz, 1, 1).normal_(0, 1)
        
        image = Image.open(self.image_files[0]).convert("RGB")
        image = self.transform(image)
        self.isize = image.shape[-1]

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file).convert("RGB")
        image = self.transform(image)
            # if image_file.parent.name == "good":
            #     target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            # else:
            #     target = Image.open(
            #         image_file.replace("/test/", "/ground_truth/").replace(
            #             ".png", "_mask.png"
            #         )
            #     )
            #     target = self.target_transform(target)
            # if self.inference:
            #     return image, image_file.stem
            # else:
        target = int(image_file.parent.name != "good")
        return image, target
        # return {'image': image, 'latentz': self.noise[index], 'index': index, 'frame_gt': target}

    def __len__(self):
        return len(self.image_files)
    
class PanoramaDataset:
    
    def __init__(self, data_dir, inference=False, train=True, cache_images=True, train_split=0.7):
        self.data_dir = Path(data_dir)
        check_paths(self.data_dir)
        if not self.data_dir.is_dir():
            raise RuntimeError(f"direcotry {str(self.data_dir.resolve())} is not a directory!")
        
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((128, 128)),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.isize = 128
        self.train = train
        self.cache_images = cache_images
        self.inference = inference
        self.image_metadata_files = [p_file for p_file in self.data_dir.iterdir() if p_file.suffix == ".xml"]
        self.image_metadata_files.sort()
        self.good_ones = []
        self.bad_ones = []
        for metadata_file in self.image_metadata_files:
            metadata = minidom.parse(metadata_file.open())
            objects = metadata.getElementsByTagName("object")
            if len(objects) <= 1:
                self.good_ones.append(metadata_file.parent / (metadata_file.stem + ".jpg"))
            else:
                self.bad_ones.append(metadata_file.parent / (metadata_file.stem + ".jpg"))
        split_idx = int(len(self.good_ones) * train_split)
        if self.train:
            # self.image_files = self.good_ones[:int(len(self.good_ones) * train_split)]
            self.dataset = [(image_file, 0) for image_file in self.good_ones[:split_idx]]
        else:
            self.dataset = [(image_file, 1) for image_file in self.bad_ones[:min(2 * split_idx, len(self.bad_ones))]]
            self.dataset += [(image_file, 0) for image_file in self.good_ones[split_idx:]]
            
            # for directory in (self.data_dir / "test").iterdir():
            #     self.image_files += list(directory.iterdir())
        random.shuffle(self.dataset)
        self.images = None
        if cache_images:
            self.images = []
            print("caching images...")
            for image_file, _ in self.dataset:
                image = Image.open(image_file).convert("RGB")
                image = self.transform(image)
                self.images.append(image)

    def __getitem__(self, index):
        image_file, target = self.dataset[index]
        if self.cache_images and self.images[index] is not None:
            image = self.images[index]
        else:
            image = Image.open(image_file).convert("RGB")
            image = self.transform(image)
        
            # if image_file.parent.name == "good":
            #     target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            # else:
            #     target = Image.open(
            #         image_file.replace("/test/", "/ground_truth/").replace(
            #             ".png", "_mask.png"
            #         )
            #     )
            #     target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.dataset)


# TODO: refactor cifar-mnist anomaly dataset functions into one generic function.

##
def get_cifar_anomaly_dataset(train_ds, valid_ds, abn_cls_idx=0):
    """[summary]
    Arguments:
        train_ds {Dataset - CIFAR10} -- Training dataset
        valid_ds {Dataset - CIFAR10} -- Validation dataset.
    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})
    Returns:
        [np.array] -- New training-test images and labels.
    """

    # Get images and labels.
    trn_img, trn_lbl = train_ds.data, np.array(train_ds.targets)
    tst_img, tst_lbl = valid_ds.data, np.array(valid_ds.targets)

    # --
    # Find idx, img, lbl for abnormal and normal on org dataset.
    nrm_trn_idx = np.where(trn_lbl != abn_cls_idx)[0]
    abn_trn_idx = np.where(trn_lbl == abn_cls_idx)[0]
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.

    nrm_tst_idx = np.where(tst_lbl != abn_cls_idx)[0]
    abn_tst_idx = np.where(tst_lbl == abn_cls_idx)[0]
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # Create new anomaly dataset based on the following data structure:
    # - anomaly dataset
    #   . -> train
    #        . -> normal
    #   . -> test
    #        . -> normal
    #        . -> abnormal
    train_ds.data = np.copy(nrm_trn_img)
    valid_ds.data = np.concatenate((nrm_tst_img, abn_trn_img, abn_tst_img), axis=0)
    train_ds.targets = np.copy(nrm_trn_lbl)
    valid_ds.targets = np.concatenate((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), axis=0)

    return train_ds, valid_ds

##
def get_mnist_anomaly_dataset(train_ds, valid_ds, abn_cls_idx=0):
    """[summary]
    Arguments:
        train_ds {Dataset - MNIST} -- Training dataset
        valid_ds {Dataset - MNIST} -- Validation dataset.
    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})
    Returns:
        [np.array] -- New training-test images and labels.
    """

    # Get images and labels.
    trn_img, trn_lbl = train_ds.data, train_ds.targets
    tst_img, tst_lbl = valid_ds.data, valid_ds.targets

    # --
    # Find normal abnormal indexes.
    # TODO: PyTorch v0.4 has torch.where function
    nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() != abn_cls_idx)[0])
    abn_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() == abn_cls_idx)[0])
    nrm_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() != abn_cls_idx)[0])
    abn_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() == abn_cls_idx)[0])

    # --
    # Find normal and abnormal images
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images.
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.

    # --
    # Find normal and abnormal labels.
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # Create new anomaly dataset based on the following data structure:
    train_ds.data = nrm_trn_img.clone()
    valid_ds.data = torch.cat((nrm_tst_img, abn_trn_img, abn_tst_img), dim=0)
    train_ds.targets = nrm_trn_lbl.clone()
    valid_ds.targets = torch.cat((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), dim=0)

    return train_ds, valid_ds

##
def make_anomaly_dataset(train_ds, valid_ds, abn_cls_idx=0):
    """[summary]

    Arguments:
        train_ds {Dataset - MNIST} -- Training dataset
        valid_ds {Dataset - MNIST} -- Validation dataset.

    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})

    Returns:
        [np.array] -- New training-test images and labels.
    """

    # Check the input type.
    if isinstance(train_ds.data, np.ndarray):
        train_ds.data = torch.from_numpy(train_ds.data)
        valid_ds.data = torch.from_numpy(valid_ds.data)
        train_ds.targets = torch.Tensor(train_ds.targets)
        valid_ds.targets = torch.Tensor(valid_ds.targets)

    # Get images and labels.
    trn_img, trn_lbl = train_ds.data, train_ds.targets
    tst_img, tst_lbl = valid_ds.data, valid_ds.targets

    # --
    # Find normal abnormal indexes.
    # TODO: PyTorch v0.4 has torch.where function
    nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() != abn_cls_idx)[0])
    abn_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() == abn_cls_idx)[0])
    nrm_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() != abn_cls_idx)[0])
    abn_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() == abn_cls_idx)[0])

    # --
    # Find normal and abnormal images
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images.
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.

    # --
    # Find normal and abnormal labels.
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # Create new anomaly dataset based on the following data structure:
    train_ds.data = nrm_trn_img.clone()
    valid_ds.data = torch.cat((nrm_tst_img, abn_trn_img, abn_tst_img), dim=0)
    train_ds.targets = nrm_trn_lbl.clone()
    valid_ds.targets = torch.cat((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), dim=0)

    return train_ds, valid_ds
