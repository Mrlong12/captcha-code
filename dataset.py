import os
from PIL import Image
from torch.utils import data
from torchvision import transforms
class Normalization(object):
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        img = sample
        img = self.normalize(img)
        return img
class ToTensor(object):
    def __init__(self):
        self.tensor = transforms.ToTensor()
    def __call__(self, sample):
        img = sample
        img = self.tensor(img)
        return img
class ImageData(data.Dataset):
    """ image dataset
    img_root:    image root (root which contain images)
    label_root:  label root (root which contains labels)
    transform:   pre-process for image
    t_transform: pre-process for label
    filename:    MSRA-B use xxx.txt to recognize train-val-test data (only for MSRA-B)
    """

    def __init__(self, path, transform_image, mode='train'):
        # os.listdir列出该目录下的所有文件名和文件夹 os.path.join路径拼接
        # 返回img_root文件夹下面的所有文件路径列表
        self.image_path = []
        img_root = ''
        img_root = path
        lines = os.listdir(path)
        self.image_path = list(map(lambda x: os.path.join(img_root, x), lines))
        self.image_path.sort(key=lambda x:int(x.split('/')[-1][:-4]))
        self.transform_image = transform_image
        self.mode = mode
    def __getitem__(self, item):
        img = Image.open(self.image_path[item])
        img = self.transform_image(img)
        return img,self.image_path[item].split('/')[-1]

    def __len__(self):
        return len(self.image_path)

def get_loader(path, batch_size, num_thread=4, pin=True, mode='train'):
    transform = transforms.Compose([
            ToTensor(),
            Normalization(),
        ]
        )
    dataset = ImageData(path, transform, mode=mode)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_thread,
                                  pin_memory=pin)
    return data_loader

