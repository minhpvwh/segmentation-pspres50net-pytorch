import cv2
from model import PSPnet
import torch
from dataset import make_datapath_list, DataTransform
from PIL import Image
import numpy as np

net = PSPnet(n_classes=21)
# Nếu muốn sử dụng CPU  map_location={'cpu', 'cpu'}
state_dict = torch.load('./pspnet50.pth', map_location={'cuda:0': 'cuda:0'})
net.load_state_dict(state_dict)
net.eval()

root_path = "F:\Pytorch\pytorch-learn\Segmentation\data\VOCdevkit\VOC2012"
train_image_list_paths, train_anno_list_paths, val_image_list_paths, val_anno_list_paths = make_datapath_list(
    root_path)
input_shape = 475
color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)

anno_file_path = val_image_list_paths[1]
anno_class_img = Image.open(anno_file_path)
p_palatte = anno_class_img.getpalette()

transform = DataTransform(input_shape, color_mean, color_std)
phase = "val"

# 0 là camera đầu tiên - có thể tùy chỉnh số lượng
cap = cv2.VideoCapture(0)

img_width, image_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while (True):
    ret, img = cap.read()
    img = Image.fromarray(img)
    img_origin = img.copy()

    img, anno_class_img = transform(phase, img, anno_class_img)
    x = img.unsqueeze(0)
    outputs = net(x)
    y = outputs[0]

    y = y[0].detach().numpy()
    y = np.argmax(y, axis=0)

    anno_class_img = Image.fromarray(np.uint8(y), mode='P')  # P là palette
    anno_class_img = anno_class_img.resize(img_width, image_height, Image.NEAREST)
    anno_class_img.putpalette(p_palatte)

    # A là độ trong suốt, RGBA có 4 channels
    trans_img = Image.new('RGBA', anno_class_img.size, (0, 0, 0, 0))
    anno_class_img = anno_class_img.convert('RGBA')

    for x in range(img_width):
        for y in range(image_height):
            pixel = anno_class_img.getpixel((x, y))
            r, g, b = pixel
            if r == 0 and g == 0 and b == 0:
                continue
            else:
                trans_img.putpixel((x, y), (r, g, b, 150))

    result = Image.alpha_composite(img_origin.convert('RGBA'), trans_img)

    # convert PIL->CV2
    img = np.array(result, dtype=np.uint8)

    # display the result frame
    cv2.imshow('PSPres50', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
