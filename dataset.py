import os.path as osp
from utils.augmentation import Compose, Scale, RandomRotation, RandomMirror, Resize, Normalize_Tensor
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt


def make_datapath_list(root_path):
    image_path = osp.join(root_path, 'JPEGImages', '%s.jpg')
    anno_path = osp.join(root_path, 'SegmentationClass', '%s.png')

    train_ids = osp.join(root_path, 'ImageSets', 'Segmentation', 'train.txt')
    anno_ids = osp.join(root_path, 'ImageSets', 'Segmentation', 'val.txt')

    train_image_list_paths = [(image_path) % path.strip() for path in open(train_ids)]
    train_anno_list_paths = [(anno_path) % path.strip() for path in open(train_ids)]

    val_image_list_paths = [(image_path) % path.strip() for path in open(anno_ids)]
    val_anno_list_paths = [(anno_path) % path.strip() for path in open(anno_ids)]

    return train_image_list_paths, train_anno_list_paths, val_image_list_paths, val_anno_list_paths


class DataTransform:
    def __init__(self, input_shape, color_mean, color_std):
        self.data_transform = {
            "train": Compose([
                Scale(scale=[0.5, 1.5]),
                RandomRotation(angle=[-10, 10]),
                RandomMirror(),
                Resize(input_shape),
                Normalize_Tensor(color_mean, color_std)
            ]),
            "val": Compose([
                Resize(input_shape),
                Normalize_Tensor(color_mean, color_std)
            ])
        }

    def __call__(self, phase, img, anno_class_img):
        return self.data_transform[phase](img, anno_class_img)


class MyDataset(data.Dataset):
    def __init__(self, phase, img_list_path, anno_list_path, transforms):
        self.phase = phase
        self.img_list_path = img_list_path
        self.anno_list_path = anno_list_path
        self.transforms = transforms

    def __len__(self):
        return len(self.img_list_path)

    def __getitem__(self, index):
        return self.pull_item(index)

    def pull_item(self, index):
        img_list_path = self.img_list_path[index]
        anno_list_path = self.anno_list_path[index]
        img = Image.open(img_list_path)
        anno_class_image = Image.open(anno_list_path)  # PIL -> (height, width, channel(RGB)

        img, anno_class_image = self.transforms(self.phase, img, anno_class_image)
        return img, anno_class_image


if __name__ == "__main__":
    root_path = "F:\Pytorch\pytorch-learn\Segmentation\data\VOCdevkit\VOC2012"
    train_image_list_paths, train_anno_list_paths, val_image_list_paths, val_anno_list_paths = make_datapath_list(
        root_path)
    print("lấy path oke")
    # image = Image.open(train_image_list_paths[0])
    # anno = Image.open(train_anno_list_paths[0])
    input_shape = 475
    color_mean = (0.485, 0.456, 0.406)
    color_std = (0.229, 0.224, 0.225)
    # data_transform = DataTransform(input_shape, color_mean, color_std)
    transform = DataTransform(input_shape, color_mean, color_std)
    train_data = MyDataset("train", train_image_list_paths, train_anno_list_paths, transform)
    val_data = MyDataset("val", val_image_list_paths, val_anno_list_paths, transform)
    print(train_data.__getitem__(0)[0])
    print(val_data.__getitem__(0)[1])
    '''
        tensor(Tensor): Tensor image of size(C, H, W) to be normalized.
    '''
    batch_size = 4
    train_dataloader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    dict_dataloader = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    batch_iter = iter(dict_dataloader["train"])
    img, anno = next(batch_iter)
    print(img.shape)  # torch.Size([4, 3, 475, 475])
    print(anno.shape)  # torch.Size([4, 475, 475])
    image = img[0]

    plt.imshow(image.numpy().transpose(1,2,0)) #(chanel(RGB), height, witdh) => (height, width, channel(RGB))
    plt.show()

    anno_class_image = anno[0].numpy()
    plt.imshow(anno_class_image)
    plt.show()