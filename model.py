import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilations, bias):
        super(conv2DBatchNormRelu, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilations, bias=bias)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(
            inplace=True)  # inplace=True giá trị của các biễn sẽ xử lý trên memory k lưu dưới thành biến khác

    # hàm forward khi đưa 1 input vào thì sẽ gọi vào hàm này
    # x là input
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        output = self.relu(x)
        return output


class FeatureMapConvolution(nn.Module):
    def __init__(self):
        print(1)
        super(FeatureMapConvolution, self).__init__()
        input_channels, output_channels, kernel_size, stride, padding, dilations, bias = 3, 64, 3, 2, 1, 1, False
        self.cbnr_1 = conv2DBatchNormRelu(input_channels, output_channels, kernel_size, stride, padding, dilations,
                                          bias)

        input_channels, output_channels, kernel_size, stride, padding, dilations, bias = 64, 64, 3, 1, 1, 1, False
        self.cbnr_2 = conv2DBatchNormRelu(input_channels, output_channels, kernel_size, stride, padding, dilations,
                                          bias)

        input_channels, output_channels, kernel_size, stride, padding, dilations, bias = 64, 128, 3, 1, 1, 1, False
        self.cbnr_3 = conv2DBatchNormRelu(input_channels, output_channels, kernel_size, stride, padding, dilations,
                                          bias)

        self.maxP = nn.MaxPool2d(kernel_size, stride=2, padding=1)

    def forward(self, x):
        x = self.cbnr_1(x)
        x = self.cbnr_2(x)
        x = self.cbnr_3(x)
        output = self.maxP(x)

        return output


class ResidualBlockPSP(nn.Sequential):
    def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation):
        print(2)
        super(ResidualBlockPSP, self).__init__()

        # bottleNeckPSP
        self.add_module("block1", bottleNeckPSP(in_channels, mid_channels, out_channels, stride, dilation))

        for i in range(n_blocks - 1):
            self.add_module("block" + str(i + 2), bottleNeckIdentifyPSP(out_channels, mid_channels, stride, dilation))


class conv2DBatchnorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(conv2DBatchnorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        output = self.bn(x)

        return output


class bottleNeckPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation):
        super(bottleNeckPSP, self).__init__()
        self.cbr_1 = conv2DBatchNormRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilations=1,
                                         bias=False)
        self.cbr_2 = conv2DBatchNormRelu(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=dilation,
                                         dilations=dilation, bias=False)
        self.cb_3 = conv2DBatchnorm(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                    bias=False)

        # skip connection
        # y = f(x) + x --  f(x) chính là tàn dư(residual)
        self.cb_residual = conv2DBatchnorm(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                           dilation=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.cb_3(self.cbr_2(self.cbr_1(x)))
        residual = self.cb_residual(x)
        return self.relu(conv + residual)


class bottleNeckIdentifyPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, stride, dilation):
        super(bottleNeckIdentifyPSP, self).__init__()

        self.cbr_1 = conv2DBatchNormRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilations=1,
                                         bias=False)
        self.cbr_2 = conv2DBatchNormRelu(mid_channels, mid_channels, kernel_size=3, stride=1, padding=dilation,
                                         dilations=dilation, bias=False)
        self.cb_3 = conv2DBatchnorm(mid_channels, in_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                    bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.cb_3(self.cbr_2(self.cbr_1(x)))
        residual = x
        return self.relu(conv + residual)


class PyramidPooling(nn.Module):
    def __init__(self, in_channel, pool_sizes, height, width):
        print(3)
        super(PyramidPooling, self).__init__()
        self.height = height
        self.width = width

        out_channel = int(in_channel / len(pool_sizes))
        # poolsize [6,3,2,1]
        self.avpool_1 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[0])
        self.cbr_1 = conv2DBatchNormRelu(in_channel, out_channel, kernel_size=1, stride=1, padding=0, dilations=1,
                                         bias=False)

        self.avpool_2 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[1])
        self.cbr_2 = conv2DBatchNormRelu(in_channel, out_channel, kernel_size=1, stride=1, padding=0, dilations=1,
                                         bias=False)

        self.avpool_3 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[2])
        self.cbr_3 = conv2DBatchNormRelu(in_channel, out_channel, kernel_size=1, stride=1, padding=0, dilations=1,
                                         bias=False)

        self.avpool_4 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[3])
        self.cbr_4 = conv2DBatchNormRelu(in_channel, out_channel, kernel_size=1, stride=1, padding=0, dilations=1,
                                         bias=False)

    def forward(self, x):
        output_1 = self.cbr_1(self.avpool_1(x))
        # UP size
        output_1 = F.interpolate(output_1, size=(self.height, self.width), mode='bilinear', align_corners=True)

        output_2 = self.cbr_2(self.avpool_2(x))
        output_2 = F.interpolate(output_2, size=(self.height, self.width), mode='bilinear', align_corners=True)

        output_3 = self.cbr_3(self.avpool_3(x))
        output_3 = F.interpolate(output_3, size=(self.height, self.width), mode='bilinear', align_corners=True)

        output_4 = self.cbr_4(self.avpool_4(x))
        output_4 = F.interpolate(output_4, size=(self.height, self.width), mode='bilinear', align_corners=True)
        return torch.cat([x, output_1, output_2, output_3, output_4], dim=1)


class DecoderModule(nn.Module):
    def __init__(self, height, width, n_classese):
        print(4)
        super(DecoderModule, self).__init__()
        self.height = height
        self.width = width
        self.cbr = conv2DBatchNormRelu(input_channels=4096, output_channels=512, kernel_size=3, stride=1, padding=1,
                                       dilations=1,
                                       bias=False)
        # lớp dropout để p= 0.1 nó sẽ loại bỏ 10% các node trong mạng 1 cách ngẫu nhiên
        self.dropout = nn.Dropout(p=0.1)
        self.classification = nn.Conv2d(in_channels=512, out_channels=n_classese, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        out = F.interpolate(x, size=(self.height, self.width), mode='bilinear', align_corners=True)
        return out


class AuxLossModule(nn.Module):
    def __init__(self, in_channels, height, width, n_classese):
        print(5)
        super(AuxLossModule, self).__init__()
        self.height = height
        self.width = width
        self.cbr = conv2DBatchNormRelu(input_channels=in_channels, output_channels=256, kernel_size=3, stride=1,
                                       padding=1,
                                       dilations=1,
                                       bias=False)
        # lớp dropout để p= 0.1 nó sẽ loại bỏ 10% các node trong mạng 1 cách ngẫu nhiên
        self.dropout = nn.Dropout(p=0.1)
        self.classification = nn.Conv2d(in_channels=256, out_channels=n_classese, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        out = F.interpolate(x, size=(self.height, self.width), mode='bilinear', align_corners=True)
        return out


class PSPnet(nn.Module):
    def __init__(self, n_classes):
        super(PSPnet, self).__init__()

        # parameters
        # Vì sử dụng
        block_config = [3, 4, 6, 3]
        image_size = 475
        image_size_8 = 60  # 6*80 ~ 475

        # Feature Module
        self.feature_conv = FeatureMapConvolution()
        self.feature_res_1 = ResidualBlockPSP(block_config[0], in_channels=128, mid_channels=64, out_channels=256,
                                              stride=1, dilation=1)
        self.feature_res_2 = ResidualBlockPSP(block_config[1], in_channels=256, mid_channels=128, out_channels=512,
                                              stride=2, dilation=1)
        self.feature_res_dilated_1 = ResidualBlockPSP(block_config[2], in_channels=512, mid_channels=256,
                                                      out_channels=1024, stride=1, dilation=2)
        self.feature_res_dilated_2 = ResidualBlockPSP(block_config[3], in_channels=1024, mid_channels=512,
                                                      out_channels=2048, stride=1, dilation=2)

        # pyramid_pooling
        self.pyramid_pooling = PyramidPooling(in_channel=2048, pool_sizes=[6, 3, 2, 1], height=image_size_8,
                                              width=image_size_8)

        # decoder module
        self.decoder = DecoderModule(height=image_size, width=image_size, n_classese=n_classes)

        # Aux
        self.aux = AuxLossModule(in_channels=1024, height=image_size, width=image_size, n_classese=n_classes)

    def forward(self, x):
        x = self.feature_conv(x)
        x = self.feature_res_1(x)
        x = self.feature_res_2(x)
        x = self.feature_res_dilated_1(x)
        out_aux = self.aux(x)
        x = self.feature_res_dilated_2(x)
        x = self.pyramid_pooling(x)
        out = self.decoder(x)
        return (out, out_aux)


if __name__ == "__main__":
    psp = PSPnet(21)
    print(psp)
    dummy_image = torch.rand(2, 3, 475, 475)
    out = psp(dummy_image)
    print(out[0].shape, out[1].shape)
