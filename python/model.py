from collections import OrderedDict
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                    padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(('relu_'+layer_name, nn.ReLU(inplace=True)))

    return nn.Sequential(OrderedDict(layers))


class GaussianFilter(nn.Module):
    def __init__(self, channel=18, pad=12, sigma=3):
        super().__init__()
        n = np.zeros((pad * 2 + 1, pad * 2 + 1))
        n[pad, pad] = 1
        kernel = torch.from_numpy(
            scipy.ndimage.gaussian_filter(n, sigma=sigma)
        ).float().requires_grad_(False)
        self.kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1)
        self.filter = nn.Conv2d(channel, channel, pad * 2 + 1, padding=0,
                                groups=channel, bias=False)
        self.filter.weight = torch.nn.Parameter(kernel)
        self.padding = nn.ReflectionPad2d(pad)

    def forward(self, x):
        x = self.padding(x)
        x = self.filter(x)
        return x

class bodypose_model(nn.Module):
    def __init__(self):
        super(bodypose_model, self).__init__()
        self.boxsize = [184]
        self.stride = 8
        self.padValue = 128
        self.thre1 = 0.1
        self.thre2 = 0.05
        self.gaussian_filter = GaussianFilter()

        # these layers have no relu layer
        no_relu_layers = ['conv5_5_CPM_L1', 'conv5_5_CPM_L2', 'Mconv7_stage2_L1',\
                          'Mconv7_stage2_L2', 'Mconv7_stage3_L1', 'Mconv7_stage3_L2',\
                          'Mconv7_stage4_L1', 'Mconv7_stage4_L2', 'Mconv7_stage5_L1',\
                          'Mconv7_stage5_L2', 'Mconv7_stage6_L1', 'Mconv7_stage6_L1']
        blocks = {}
        block0 = OrderedDict({'conv1_1': [3, 64, 3, 1, 1],
                  'conv1_2': [64, 64, 3, 1, 1],
                  'pool1_stage1': [2, 2, 0],
                  'conv2_1': [64, 128, 3, 1, 1],
                  'conv2_2': [128, 128, 3, 1, 1],
                  'pool2_stage1': [2, 2, 0],
                  'conv3_1': [128, 256, 3, 1, 1],
                  'conv3_2': [256, 256, 3, 1, 1],
                  'conv3_3': [256, 256, 3, 1, 1],
                  'conv3_4': [256, 256, 3, 1, 1],
                  'pool3_stage1': [2, 2, 0],
                  'conv4_1': [256, 512, 3, 1, 1],
                  'conv4_2': [512, 512, 3, 1, 1],
                  'conv4_3_CPM': [512, 256, 3, 1, 1],
                  'conv4_4_CPM': [256, 128, 3, 1, 1]})

        # Stage 1
        block1_1 = OrderedDict({'conv5_1_CPM_L1': [128, 128, 3, 1, 1],
                    'conv5_2_CPM_L1': [128, 128, 3, 1, 1],
                    'conv5_3_CPM_L1': [128, 128, 3, 1, 1],
                    'conv5_4_CPM_L1': [128, 512, 1, 1, 0],
                    'conv5_5_CPM_L1': [512, 38, 1, 1, 0]})

        block1_2 = OrderedDict({'conv5_1_CPM_L2': [128, 128, 3, 1, 1],
                    'conv5_2_CPM_L2': [128, 128, 3, 1, 1],
                    'conv5_3_CPM_L2': [128, 128, 3, 1, 1],
                    'conv5_4_CPM_L2': [128, 512, 1, 1, 0],
                    'conv5_5_CPM_L2': [512, 19, 1, 1, 0]})
        blocks['block1_1'] = block1_1
        blocks['block1_2'] = block1_2

        self.model0 = make_layers(block0, no_relu_layers)

        # Stages 2 - 6
        for i in range(2, 7):
            blocks['block%d_1' % i] = OrderedDict({
                'Mconv1_stage%d_L1' % i: [185, 128, 7, 1, 3],
                'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3],
                'Mconv3_stage%d_L1' % i: [128, 128, 7, 1, 3],
                'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3],
                'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3],
                'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0],
                'Mconv7_stage%d_L1' % i: [128, 38, 1, 1, 0]})

            blocks['block%d_2' % i] = OrderedDict({
                'Mconv1_stage%d_L2' % i: [185, 128, 7, 1, 3],
                'Mconv2_stage%d_L2' % i: [128, 128, 7, 1, 3],
                'Mconv3_stage%d_L2' % i: [128, 128, 7, 1, 3],
                'Mconv4_stage%d_L2' % i: [128, 128, 7, 1, 3],
                'Mconv5_stage%d_L2' % i: [128, 128, 7, 1, 3],
                'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0],
                'Mconv7_stage%d_L2' % i: [128, 19, 1, 1, 0]})

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_1 = blocks['block1_1']
        self.model2_1 = blocks['block2_1']
        self.model3_1 = blocks['block3_1']
        self.model4_1 = blocks['block4_1']
        self.model5_1 = blocks['block5_1']
        self.model6_1 = blocks['block6_1']

        self.model1_2 = blocks['block1_2']
        self.model2_2 = blocks['block2_2']
        self.model3_2 = blocks['block3_2']
        self.model4_2 = blocks['block4_2']
        self.model5_2 = blocks['block5_2']
        self.model6_2 = blocks['block6_2']

    def normalize_img(self, x):
        shape_pymd = [(bs, self.stride * (x.size(3) / x.size(2) * bs) // self.stride)
                 for bs in self.boxsize]
        x = [F.interpolate(x, shape, mode="bilinear") / 256 - 0.5 for shape in shape_pymd]
        return x

    def forward(self, x):
        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)

        heatmap = self.model6_1(out6)
        paf = self.model6_2(out6)
        return heatmap, paf

    def forward_parallel(self, x):
        b, c, h, w = x.shape
        heatmap_avg = torch.zeros(c, h, w, 19)
        paf_avg = torch.zeros(c, h, w, 38)
        img_pyramid = self.normalize_img(x)

        for x in img_pyramid:
            out1 = self.model0(x)

            out1_1 = self.model1_1(out1)
            out1_2 = self.model1_2(out1)
            out2 = torch.cat([out1_1, out1_2, out1], 1)

            out2_1 = self.model2_1(out2)
            out2_2 = self.model2_2(out2)
            out3 = torch.cat([out2_1, out2_2, out1], 1)

            out3_1 = self.model3_1(out3)
            out3_2 = self.model3_2(out3)
            out4 = torch.cat([out3_1, out3_2, out1], 1)

            out4_1 = self.model4_1(out4)
            out4_2 = self.model4_2(out4)
            out5 = torch.cat([out4_1, out4_2, out1], 1)

            out5_1 = self.model5_1(out5)
            out5_2 = self.model5_2(out5)
            out6 = torch.cat([out5_1, out5_2, out1], 1)

            heatmap = self.model6_1(out6)
            paf = self.model6_2(out6)

            heatmap = F.interpolate(heatmap, (h, w), mode="bilinear")
            paf = F.interpolate(paf, (h, w), mode="bilinear")

            heatmap_avg += heatmap_avg + heatmap / len(img_pyramid)
            paf_avg += paf / len(img_pyramid)

        one_heatmap = self.gaussian_filter(heatmap_avg)
        map_left = np.zeros(one_heatmap.shape)
        map_left[:, :, 1:, :] = one_heatmap[:, :, :-1, :]
        map_right = np.zeros(one_heatmap.shape)
        map_right[:, :, :-1, :] = one_heatmap[:, :, 1:, :]
        map_up = np.zeros(one_heatmap.shape)
        map_up[:, :, :, 1:] = one_heatmap[:, :, :, :-1]
        map_down = np.zeros(one_heatmap.shape)
        map_down[:, :, :, :-1] = one_heatmap[:, :, :, 1:]

        peaks_binary = one_heatmap >= map_left * one_heatmap >= map_right * \
                       one_heatmap >= map_up * one_heatmap >= map_down * \
                       one_heatmap > self.thre1
        peaks = peaks_binary.nonzero()


        for part in range(18):
            map_ori = heatmap_avg[:, :, part]
            one_heatmap = gaussian_filter(map_ori, sigma=3)



            peaks_binary = np.logical_and.reduce(
                (one_heatmap >= map_left, one_heatmap >= map_right, one_heatmap >= map_up, one_heatmap >= map_down, one_heatmap > thre1))
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            peak_id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        return out6_1, out6_2

class handpose_model(nn.Module):
    def __init__(self):
        super(handpose_model, self).__init__()

        # these layers have no relu layer
        no_relu_layers = ['conv6_2_CPM', 'Mconv7_stage2', 'Mconv7_stage3',\
                          'Mconv7_stage4', 'Mconv7_stage5', 'Mconv7_stage6']
        # stage 1
        block1_0 = OrderedDict({
            'conv1_1': [3, 64, 3, 1, 1],
            'conv1_2': [64, 64, 3, 1, 1],
            'pool1_stage1': [2, 2, 0],
            'conv2_1': [64, 128, 3, 1, 1],
            'conv2_2': [128, 128, 3, 1, 1],
            'pool2_stage1': [2, 2, 0],
            'conv3_1': [128, 256, 3, 1, 1],
            'conv3_2': [256, 256, 3, 1, 1],
            'conv3_3': [256, 256, 3, 1, 1],
            'conv3_4': [256, 256, 3, 1, 1],
            'pool3_stage1': [2, 2, 0],
            'conv4_1': [256, 512, 3, 1, 1],
            'conv4_2': [512, 512, 3, 1, 1],
            'conv4_3': [512, 512, 3, 1, 1],
            'conv4_4': [512, 512, 3, 1, 1],
            'conv5_1': [512, 512, 3, 1, 1],
            'conv5_2': [512, 512, 3, 1, 1],
            'conv5_3_CPM': [512, 128, 3, 1, 1]})

        block1_1 = OrderedDict({
            'conv6_1_CPM': [128, 512, 1, 1, 0],
            'conv6_2_CPM': [512, 22, 1, 1, 0]
        })

        blocks = {}
        blocks['block1_0'] = block1_0
        blocks['block1_1'] = block1_1

        # stage 2-6
        for i in range(2, 7):
            blocks['block%d' % i] = OrderedDict({
                'Mconv1_stage%d' % i: [150, 128, 7, 1, 3],
                'Mconv2_stage%d' % i: [128, 128, 7, 1, 3],
                'Mconv3_stage%d' % i: [128, 128, 7, 1, 3],
                'Mconv4_stage%d' % i: [128, 128, 7, 1, 3],
                'Mconv5_stage%d' % i: [128, 128, 7, 1, 3],
                'Mconv6_stage%d' % i: [128, 128, 1, 1, 0],
                'Mconv7_stage%d' % i: [128, 22, 1, 1, 0]})

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_0 = blocks['block1_0']
        self.model1_1 = blocks['block1_1']
        self.model2 = blocks['block2']
        self.model3 = blocks['block3']
        self.model4 = blocks['block4']
        self.model5 = blocks['block5']
        self.model6 = blocks['block6']

    def forward(self, x):
        out1_0 = self.model1_0(x)
        out1_1 = self.model1_1(out1_0)
        concat_stage2 = torch.cat([out1_1, out1_0], 1)
        out_stage2 = self.model2(concat_stage2)
        concat_stage3 = torch.cat([out_stage2, out1_0], 1)
        out_stage3 = self.model3(concat_stage3)
        concat_stage4 = torch.cat([out_stage3, out1_0], 1)
        out_stage4 = self.model4(concat_stage4)
        concat_stage5 = torch.cat([out_stage4, out1_0], 1)
        out_stage5 = self.model5(concat_stage5)
        concat_stage6 = torch.cat([out_stage5, out1_0], 1)
        out_stage6 = self.model6(concat_stage6)
        return out_stage6


