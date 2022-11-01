"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor, GoogLeNet, U_Net
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention
import torch
import torchvision.models as models
import torch.nn as nn



from efficientnet_pytorch import EfficientNet

class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'GoogLeNet':
            # self.FeatureExtraction = GoogLeNet(opt.input_channel, opt.output_channel)
            model = models.googlenet(pretrained=True)
            model.conv1.conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # model.inception5b.branch4[1].conv = nn.Conv2d(832, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            # model.inception5b.branch4[1].bn = nn.BatchNorm2d(512, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
            model = torch.nn.Sequential(*(list(model.children())[:-1]))
            layer1 = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            layer2 = nn.BatchNorm2d(512, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
            model = nn.Sequential(model, layer1, layer2)
            self.FeatureExtraction = model
        elif opt.FeatureExtraction == 'nvidia_efficientNet_widese_b0':
            model_1 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_widese_b0', pretrained=True)
            model_1 = torch.nn.Sequential(*(list(model_1.children())[:-1]))

            layer_1 = nn.Conv2d(1280, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            layer_2 = nn.BatchNorm2d(512, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
            layer_3 = nn.SiLU(inplace=True)

            self.FeatureExtraction = nn.Sequential(model_1, layer_1, layer_2, layer_3)
        elif opt.FeatureExtraction == 'nvidia_efficientNet_widese_b0_1':
            model_1 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_widese_b0',
                                     pretrained=True)
            model_1 = torch.nn.Sequential(*(list(model_1.children())[:-1]))

            layer_1 = nn.Conv2d(1280, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            layer_2 = nn.BatchNorm2d(512, eps=0.001, momentum=0.010000000000000009, affine=True,
                                     track_running_stats=True)
            layer_3 = nn.SiLU(inplace=True)

            self.FeatureExtraction = nn.Sequential(model_1, layer_1, layer_2, layer_3)

        elif opt.FeatureExtraction == 'UNet':
            self.FeatureExtraction = U_Net(opt.input_channel, opt.output_channel)
            # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ FeatureExtranction @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@') batch x 512 x h x w 512  1  4
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        # print('@@@@@@@@@@@@@@@@@@@@@@ Sequence MOdleing @@@@@@@@@@@@@@@@@@@@@@')
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.SequenceModeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        # print('@@@@@@@@@@@@@@@@@@@@@@ Prediction @@@@@@@@@@@@@@@@@@@@@@')
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(opt.num_class, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        # print('@@@@@@@@@@@@@@@@@@@@@@ Transformation Stage @@@@@@@@@@@@@@@@@@@@@@')
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        # print('@@@@@@@@@@@@@@@@@@@@@@ Feature extraction stage @@@@@@@@@@@@@@@@@@@@@@')

        visual_feature = self.FeatureExtraction(input)
        # print('########## visual Feature ###########')
        # visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        # visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        # print('@@@@@@@@@@@@@@@@@@@@@@ Sequence modeling stage @@@@@@@@@@@@@@@@@@@@@@')
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        # print('@@@@@@@@@@@@@@@@@@@@@@ Prediction stage @@@@@@@@@@@@@@@@@@@@@@')
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return prediction
