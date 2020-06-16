from model import PSPnet
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import math
from dataset import make_datapath_list, DataTransform, MyDataset
import torch.utils.data as data
import time

# model
model = PSPnet(n_classes=21)


# loss
class PSPloss(nn.Module):
    def __init__(self, aux_weight=0.4):
        super(PSPloss, self).__init__()
        self.aux_weight = aux_weight

    def forward(self, outputs, targets):
        loss = F.cross_entropy(outputs[0], targets, reduction='mean')
        loss_aux = F.cross_entropy(outputs[1], targets, reduction='mean')
        return loss + self.aux_weight * loss_aux


criterion = PSPloss(aux_weight=0.4)

# optimizer
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# kiểu viết này phù hợp với việc fine-tuning
optimizer = optim.SGD([
    {'params': model.feature_conv.parameters(), 'lr': 1e-3},
    {'params': model.feature_res_1.parameters(), 'lr': 1e-3},
    {'params': model.feature_res_2.parameters(), 'lr': 1e-3},
    {'params': model.feature_res_dilated_1.parameters(), 'lr': 1e-3},
    {'params': model.feature_res_dilated_2.parameters(), 'lr': 1e-3},
    {'params': model.pyramid_pooling.parameters(), 'lr': 1e-3},
    {'params': model.decoder.parameters(), 'lr': 1e-2},
    {'params': model.aux.parameters(), 'lr': 1e-2},
], momentum=0.9, weight_decay=0.0001)


# scheduler
def lambda_epoch(epoch):
    max_epoch = 30
    return math.pow(1 - epoch / max_epoch, 0.9)


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)


# train_model
def train_model(model, dataloader_dict, criterion, scheduler, optimizer, num_epochs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    # tối ưu quá trình và tăng tốc train
    torch.backends.cudnn.benchmark = True

    num_train_imgs = len(dataloader_dict["train"].dataset)
    num_val_imgs = len(dataloader_dict["val"].dataset)
    batch_size = dataloader_dict["train"].batch_size

    iteration = 1
    logs = []

    # mỗi lần đi qua 1 ảnh cập nhật parameter thì rất nhỏ lẻ nên ta quy định bn ảnh (batch_multiplaier = 3 thì là 4*3=12 ảnh - batch_size = 4)thì cập nhật lại parameter
    batch_multiplaier = 3

    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        t_iter_start = time.time()
        epoch_train_loss = 0
        epoch_val_loss = 0

        print('Epoch {} / {}'.format(epoch + 1, num_epochs))
        for phase in ["train", "val"]:
            if phase == "train":
                # chuyển model về mode train để có thể cập nhật các parameter
                model.train()
                scheduler.step()
                optimizer.zero_grad()
                print("this is train")
            else:
                # cứ 5 epochs đánh giá 1 lần
                if ((epoch + 1) % 5) == 0:
                    model.eval()
                    print('this is val')
                else:
                    continue

            count = 0
            for images, anno_class_images in dataloader_dict[phase]:
                print(images.size()[0])
                if images.size()[0] == 1:
                    continue
                images = images.to(device)
                anno_class_images = anno_class_images.to(device)

                if phase == 'train' and count == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    count = batch_multiplaier

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(images)
                    # tính loss trung bình của 3 lần
                    loss = criterion(outputs, anno_class_images.long()) / batch_multiplaier

                    if phase == 'train':
                        loss.backward()
                        count -= 1

                        if iteration % 10 == 0:
                            t_iter_end = time.time()
                            duration = t_iter_end - t_iter_start
                            print('Iteration {} || Loss: {:.6f}  ||  10iter: {:.6f} sec'.format(iteration,
                                                                                                loss.item() / batch_size * batch_multiplaier,
                                                                                                duration))

                            t_iter_start = time.time()

                        epoch_train_loss += loss.item() * batch_multiplaier
                        iteration += 1
                    else:
                        epoch_val_loss += loss.item() * batch_multiplaier
        t_epoch_end = time.time()
        duration = t_epoch_end
        print('Epoch {} || Epoch_train_loss: {:.6f} || Epoch_val_loss: {:.6f}'.format(epoch + 1,
                                                                                      epoch_train_loss / num_train_imgs,
                                                                                      epoch_train_loss / num_val_imgs))
        print('Duration {:.6f} sec'.format(duration))
        t_epoch_start = time.time()

        # Nếu lưu theo từng epoch
        # torch.save(model.state_dict(), 'pspnet50.pth')

    torch.save(model.state_dict(), 'pspnet50_' + str(epoch) + '.pth')


if __name__ == "__main__":
    num_epochs = 100

    # dataloader_dict
    root_path = "F:\Pytorch\pytorch-learn\Segmentation\data\VOCdevkit\VOC2012"
    train_image_list_paths, train_anno_list_paths, val_image_list_paths, val_anno_list_paths = make_datapath_list(
        root_path)
    print("lấy path oke")

    input_shape = 475
    color_mean = (0.485, 0.456, 0.406)
    color_std = (0.229, 0.224, 0.225)
    transform = DataTransform(input_shape, color_mean, color_std)

    train_data = MyDataset("train", train_image_list_paths, train_anno_list_paths, transform)
    val_data = MyDataset("val", val_image_list_paths, val_anno_list_paths, transform)

    # batch_size = 4
    batch_size = 1
    train_dataloader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    dict_dataloader = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    train_model(model, dict_dataloader, criterion, scheduler, optimizer, num_epochs=num_epochs)
