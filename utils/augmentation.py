import numpy as np
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms
import torch


class Compose:
    def __init__(self, transforms):
        self.trasnforms = transforms

    def __call__(self, img, anno_class_img):
        for t in self.trasnforms:
            img, anno_class_img = t(img, anno_class_img)
        return img, anno_class_img


class Scale:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img, anno_class_img):
        img_w = img.size[0]
        img_h = img.size[1]
        # print(img_w, img_h)

        scale = np.random.uniform(self.scale[0], self.scale[1])

        scale_w = int(scale * img_w)
        scale_h = int(scale * img_h)

        anno_class_img = anno_class_img.resize((scale_w, scale_h), Image.NEAREST)
        img = img.resize((scale_w, scale_h), Image.BICUBIC)

        if scale > 1.0:
            left = scale_w - img_w
            top = scale_h - img_h

            left = int(np.random.uniform(0, left))
            top = int(np.random.uniform(0, top))

            img = img.crop((left, top, left + img_w, top + img_h))
            anno_class_img = anno_class_img.crop((left, top, left + img_w, top + img_h))
        else:
            # khác vs ảnh RGB thì ảnh pallet có 1 kênh màu thoi
            # lấy kênh màu của ảnh pallet
            p_palete = anno_class_img.copy().getpalette()
            img_original = img.copy()
            anno_class_img_original = anno_class_img.copy()

            left = img_w - scale_w
            left = int(np.random.uniform(0, left))

            top = img_h - scale_h
            top = int(np.random.uniform(0, top))

            img = Image.new(img.mode, (img_w, img_h), (0, 0, 0))
            img.paste(img_original, (left, top))

            anno_class_img = Image.new(anno_class_img.mode, (img_w, img_h), (0))
            anno_class_img.paste(anno_class_img_original, (left, top))
            anno_class_img.putpalette(p_palete)
        return img, anno_class_img


class RandomRotation:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img, anno_class_img):
        rotate_angle = np.random.uniform(self.angle[0], self.angle[1])

        img = img.rotate(rotate_angle, Image.BILINEAR)
        anno_class_img = anno_class_img.rotate(rotate_angle, Image.NEAREST)

        return img, anno_class_img


class RandomMirror:
    def __call__(self, img, anno_class_img):
        if np.random.randint(2):
            img = ImageOps.mirror(img)
            anno_class_img = ImageOps.mirror(anno_class_img)
        return img, anno_class_img


class Resize:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def __call__(self, img, anno_class_img):
        img = img.resize((self.input_shape, self.input_shape), Image.BILINEAR)
        anno_class_img = anno_class_img.resize((self.input_shape, self.input_shape), Image.NEAREST)
        return img, anno_class_img


class Normalize_Tensor:
    def __init__(self, color_mean, color_std):
        self.color_mean = color_mean
        self.color_std = color_std

    def __call__(self, img, anno_class_img):
        img = transforms.functional.to_tensor(img)
        img = transforms.functional.normalize(img, self.color_mean, self.color_std)

        anno_class_img = np.array(anno_class_img)
        # trong phần ảnh palette có viền trắng quanh đối tượng là nhũng phần k chắc chắn được gọi là ambigious
        # h đưa về dạng 0(black)
        # tìm những phần màu trắng là các pixel màu trắng
        index = np.where(anno_class_img == 255)
        anno_class_img[index] = 0

        # đưa lại về torch
        anno_class_img = torch.from_numpy(anno_class_img)
        return img, anno_class_img
