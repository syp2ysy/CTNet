# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg import utils
from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers
from paddle.fluid.layers import transpose

@manager.MODELS.add_component
class CTNet(nn.Layer):
    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices,
                 mid_channels=512,
                 key_channels=256,
                 align_corners=False,
                 pretrained=None):
        super().__init__()

        self.backbone = backbone
        self.backbone_indices = backbone_indices
        in_channels = [self.backbone.feat_channels[i] for i in backbone_indices]

        self.head = CTHead(
            num_classes=num_classes,
            in_channels=in_channels,
            mid_channels=mid_channels,
            key_channels=key_channels)

        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        feats = self.backbone(x)
        feats = [feats[i] for i in self.backbone_indices]
        logit_list, cls_app = self.head(feats)
        if not self.training:
            logit_list = [logit_list[0]]

        logit_list = [
            F.interpolate(
                logit,
                paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]
        a = cls_app.numpy()
        logit_list.append(cls_app)
        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class CTHead(nn.Layer):
    def __init__(self,
                 num_classes,
                 in_channels,
                 mid_channels=512,
                 key_channels=256):
        super().__init__()

        self.num_classes = num_classes
        self.ccm = CCM(mid_channels, num_classes)
        self.scm = SCM(mid_channels, key_channels, mid_channels)
        self.indices = [-2, -1] if len(in_channels) > 1 else [-1, -1]

        self.conv3x3 = layers.ConvBNReLU(
            in_channels[self.indices[1]], mid_channels, 3, padding=1)  #self.conv3x3_ocr
        self.cls_head = nn.Conv2D(mid_channels, self.num_classes, 1)
        self.aux_head = nn.Sequential(
            layers.ConvBNReLU(in_channels[self.indices[0]],
                              in_channels[self.indices[0]], 1),
            nn.Conv2D(in_channels[self.indices[0]], self.num_classes, 1))

        self.init_weight()

    def forward(self, feat_list):
        feat_shallow, feat_deep = feat_list[self.indices[0]], feat_list[
            self.indices[1]]

        soft_regions = self.aux_head(feat_shallow)
        pixels = self.conv3x3(feat_deep)

        context, cls_app, cls_matrix = self.ccm(pixels)
        scm_f = self.scm(context, cls_matrix)
        cls_app = cls_app.squeeze(-1)
        logit = self.cls_head(scm_f)
        return [logit, soft_regions], cls_app

    def init_weight(self):
        
        for sublayer in self.sublayers():
            if isinstance(sublayer, nn.Conv2D):
                param_init.normal_init(sublayer.weight, std=0.001)
            elif isinstance(sublayer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(sublayer.weight, value=1.0)
                param_init.constant_init(sublayer.bias, value=0.0)
		
class ccm_layer(nn.Layer):
    def __init__(self, k_size=[9, 17, 33, 65]):
        super(ccm_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)

        self.conv0 = nn.Conv1D(1, 1, kernel_size=k_size[0], padding=(k_size[0] - 1) // 2, bias_attr=False)
        self.conv1 = nn.Conv1D(1, 1, kernel_size=k_size[1], padding=(k_size[1] - 1) // 2, bias_attr=False)
        self.conv2 = nn.Conv1D(1, 1, kernel_size=k_size[2], padding=(k_size[2] - 1) // 2, bias_attr=False)
        self.conv3 = nn.Conv1D(1, 1, kernel_size=k_size[3], padding=(k_size[3] - 1) // 2, bias_attr=False)
        self.sigmoid = nn.Sigmoid()
        self.line = nn.Linear(4, 1, bias_attr=False)
    def forward(self, x, k_size):
        y = transpose(self.avg_pool(x).squeeze(-1),perm=[0,2,1])
        y0 = transpose(self.conv0(y), perm=[0,2,1]).unsqueeze(-1)
        y1 = transpose(self.conv1(y), perm=[0,2,1]).unsqueeze(-1)
        y3 = transpose(self.conv2(y), perm=[0,2,1]).unsqueeze(-1)
        y4 = transpose(self.conv3(y), perm=[0,2,1]).unsqueeze(-1)
        y_full = self.line(paddle.concat([y0, y1, y3, y4], axis=2).squeeze(-1).squeeze(-1)).unsqueeze(-1)
        y = self.sigmoid(y_full)

        return x * y.expand_as(x), y

class CCM(nn.Layer):

    def __init__(self, pixels_channels, class_channels):
        super().__init__()
        self.pixels_channels = pixels_channels
        self.class_channels = class_channels
        self.ccm_layer = ccm_layer(k_size=[9,17,33,65])
        self.fc1 = nn.Sequential(
			layers.ConvBN(pixels_channels, class_channels, kernel_size=1),
			nn.Softmax(axis=1))
    def forward(self, pixels):
        # pixels: from (n, c, h, w) to (n, h*w, c)
        ccm_feat, channel_context = self.ccm_layer(pixels, [9, 17, 33, 65])
        cls_app = self.fc1(channel_context).squeeze(3)
        cls_matrix = paddle.bmm(channel_context.squeeze(3), cls_app.transpose([0, 2, 1])).unsqueeze(3)
        cls_matrix = nn.functional.relu(cls_matrix)
        return ccm_feat, cls_app, cls_matrix


class SCM(nn.Layer):

    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 dropout_rate=0):
        super().__init__()

        self.scm_block = scm_layer(in_channels, key_channels)
        self.conv3x3 = nn.Sequential(
            layers.ConvBNReLU(in_channels, out_channels, 3, 1)
        )
        self.conv1x1 = nn.Sequential(
            layers.ConvBNReLU(2 * in_channels, out_channels, 1),
            nn.Dropout2D(dropout_rate))

    def forward(self, pixels, cls_matrix):
        context = self.scm_block(pixels, cls_matrix)
        pixels = self.conv3x3(pixels)
        feats = paddle.concat([context, pixels], axis=1)
        feats = self.conv1x1(feats)

        return feats


class scm_layer(nn.Layer):
    def __init__(self, in_channels, key_channels):
        super().__init__()

        self.in_channels = in_channels
        self.key_channels = key_channels
	# or use --self.f_pixel = layers.ConvBNReLU(in_channels, key_channels, 1) 
	# or use --self.f_pixel = nn.Sequential(layers.ConvBNReLU(in_channels, key_channels, 1),layers.ConvBNReLU(key_channels, key_channels, 1))
        self.f_query = layers.ConvBNReLU(in_channels, key_channels, 3, 1)  
	
	# or use --self.f_object = nn.Sequential(layers.ConvBNReLU(in_channels, key_channels, 1),layers.ConvBNReLU(key_channels, key_channels, 1))
        self.f_key = layers.ConvBNReLU(in_channels, key_channels, 1)  

        self.f_value = layers.ConvBNReLU(in_channels, key_channels, 1)

        self.f_up = layers.ConvBNReLU(key_channels, in_channels, 1)
	
        self.fuse = layers.ConvBNReLU(in_channels, in_channels, 3, 1)
    def forward(self, x, proxy):
        x_shape = paddle.shape(x)
     
        query = self.f_query(x)
        query = paddle.reshape(query, (0, self.key_channels, -1))
        query = paddle.transpose(query, (0, 2, 1))

        key = self.f_key(proxy)
        key = paddle.reshape(key, (0, self.key_channels, -1))

        value = self.f_value(proxy)
        value = paddle.reshape(value, (0, self.key_channels, -1))
        value = paddle.transpose(value, (0, 2, 1))

        sim_map = paddle.bmm(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, axis=-1)

        context = paddle.bmm(sim_map, value)
        context = paddle.transpose(context, (0, 2, 1))
        context = paddle.reshape(context, (0, self.key_channels, x_shape[2], x_shape[3]))
        context = self.f_up(context)
        context = self.fuse(context + x)
        return context
