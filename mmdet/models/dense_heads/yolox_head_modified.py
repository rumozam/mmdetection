# Copyright (c) OpenMMLab. All rights reserved.
import math
import pickle as pkl
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule,
                      bias_init_with_prob)
from mmcv.ops.nms import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import (MlvlPointGenerator, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.losses import HingeEmbeddingLoss
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin


@HEADS.register_module()
class YOLOXHeadModified(BaseDenseHead, BBoxTestMixin):
    """Modified version of YOLOXHead head used in `YOLOX <https://arxiv.org/abs/2107.08430>`_.
    Modified such that it actually works with softmax.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels in stacking convs.
            Default: 256
        stacked_convs (int): Number of stacking convs of the head.
            Default: 2.
        strides (tuple): Downsample factor of each feature map.
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        dcn_on_last_conv (bool): If true, use dcn in the last layer of
            towers. Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer. Default: None.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_obj (dict): Config of objectness loss.
        loss_l1 (dict): Config of L1 loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=2,
                 strides=[8, 16, 32],
                 use_depthwise=False,
                 dcn_on_last_conv=False,
                 conv_bias='auto',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='IoULoss',
                     mode='square',
                     eps=1e-16,
                     reduction='sum',
                     loss_weight=5.0),
                 loss_obj=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     loss_weight=1.0),
                 use_l1=False,
                 loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 use_zsd=True,
                 use_gzsd=False,
                 used_split="65-15",
                 use_tuned_imagenet_labels=True,
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu'),
                 cls_out_channels=512):

        super().__init__(init_cfg=init_cfg)

        assert cls_out_channels == 512 or cls_out_channels == 768,\
            "yolox_head_modified only support cls_out_channels == 512 or 768"

        if cls_out_channels == 512:
            # load label embeddings matrix for COCO
            # the matrix is of size 512x80, i.e. each column contains one embedding
            # the embeddings are already normed to length 1
            with open('mmdet/label_embeddings_coco.pkl', 'rb') as handle:
                self.label_embeddings_coco = pkl.load(handle).to("cuda")
                print("label embeddings for COCO loaded...")

            # load label embeddings matrix for Imagenet
            # the matrix is of size 512x1000, i.e. each column contains one embedding
            # the embeddings are already normed to length 1
            if use_tuned_imagenet_labels:
                with open('mmdet/label_embeddings_imagenet_tuned_vitb32.pkl', 'rb') as handle:
                    self.label_embeddings_imagenet = pkl.load(handle).to("cuda")
                    print("label embeddings for imagenet loaded...")
            else:
                with open('mmdet/label_embeddings_imagenet_untuned_vitb32.pkl', 'rb') as handle:
                    self.label_embeddings_imagenet = pkl.load(handle).to("cuda")
                    print("label embeddings for imagenet loaded...")
        
        else:
            # load label embeddings matrix for COCO
            # the matrix is of size 768x80, i.e. each column contains one embedding
            # the embeddings are already normed to length 1
            with open('mmdet/label_embeddings_coco_vitl14_336px.pkl', 'rb') as handle:
                self.label_embeddings_coco = pkl.load(handle).to("cuda")
                print("label embeddings for COCO loaded...")

            # load label embeddings matrix for Imagenet
            # the matrix is of size 768x1000, i.e. each column contains one embedding
            # the embeddings are already normed to length 1
            if use_tuned_imagenet_labels:
                with open('mmdet/label_embeddings_imagenet_tuned_vitl14_336px.pkl', 'rb') as handle:
                    self.label_embeddings_imagenet = pkl.load(handle).to("cuda")
                    print("label embeddings for imagenet loaded...")
            else:
                with open('mmdet/label_embeddings_imagenet_untuned_vitl14_336px.pkl', 'rb') as handle:
                    self.label_embeddings_imagenet = pkl.load(handle).to("cuda")
                    print("label embeddings for imagenet loaded...")

        # for Imagenet data ious can not be calculated since we don't have bounding boxes
        # we assume iuo = 0.3 for Imagenet images (see YOLO9000 paper)
        self.pos_ious = torch.tensor([0.3]).to('cuda')

        self.num_classes = num_classes
        self.cls_out_channels = cls_out_channels
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.use_depthwise = use_depthwise
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.use_sigmoid_cls = True

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_obj = build_loss(loss_obj)

        self.use_l1 = use_l1  # This flag will be modified by hooks.
        self.loss_l1 = build_loss(loss_l1)

        self.prior_generator = MlvlPointGenerator(strides, offset=0)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.fp16_enabled = False

        self.use_zsd = use_zsd
        self.use_gzsd = use_gzsd
        assert not self.use_zsd or not self.use_gzsd, \
            "decide whether model should predict seen classes or not"

        # same initialization as in CLIP paper
        self.logit_scale = nn.Parameter(torch.ones([]) * (1/0.07))

        assert used_split == "65-15" or used_split == "48-17", \
            "yolox_head_modified only supports used_split = '65-15' or '48-17'"

        # indices where generated by DLCV/COCO/get_indices_for_unseen_classes.py
        if used_split == "65-15":
            self.unknown_indices_coco = torch.Tensor([4, 6, 12, 15, 21, 28, 29, 31, 42, 48, 52, 61, 64, 70, 78])\
                .to(torch.int64).to('cuda')
            self.known_indices_coco = torch.Tensor([i for i in np.arange(80) if i not in self.unknown_indices_coco])\
                .to(torch.int64).to('cuda')
        else:
            self.unknown_indices_coco = torch.Tensor([4, 5, 15, 16, 19, 20, 25, 27, 31, 36, 41, 43, 55, 57, 66, 71, 76])\
                .to(torch.int64).to('cuda')
            self.known_indices_coco = torch.Tensor([0, 1, 2, 3, 6, 7, 8, 13, 14, 17, 18, 21, 22, 23, 24, 26, 28, 29, 30,\
                33, 37, 39, 42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 56, 59, 61, 62, 63, 64, 65, 68, 69, 70, 72, 73, 74, 75, 79])\
                .to(torch.int64).to('cuda')

        self.label_id_to_seen_index = self._get_label_to_index_list().to('cuda')

        self._init_layers()

    def _init_layers(self):
        self.multi_level_cls_convs = nn.ModuleList()
        self.multi_level_reg_convs = nn.ModuleList()
        self.multi_level_conv_cls = nn.ModuleList()
        self.multi_level_conv_reg = nn.ModuleList()
        self.multi_level_conv_obj = nn.ModuleList()
        for _ in self.strides:
            self.multi_level_cls_convs.append(self._build_stacked_convs())
            self.multi_level_reg_convs.append(self._build_stacked_convs())
            conv_cls, conv_reg, conv_obj = self._build_predictor()
            self.multi_level_conv_cls.append(conv_cls)
            self.multi_level_conv_reg.append(conv_reg)
            self.multi_level_conv_obj.append(conv_obj)

    def _build_stacked_convs(self):
        """Initialize conv layers of a single level head."""
        conv = DepthwiseSeparableConvModule \
            if self.use_depthwise else ConvModule
        stacked_convs = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            stacked_convs.append(
                conv(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    bias=self.conv_bias))
        return nn.Sequential(*stacked_convs)

    def _build_predictor(self):
        """Initialize predictor layers of a single level head."""
        conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        conv_reg = nn.Conv2d(self.feat_channels, 4, 1)
        conv_obj = nn.Conv2d(self.feat_channels, 1, 1)
        return conv_cls, conv_reg, conv_obj

    def _get_label_to_index_list(self):
        label_to_index = torch.zeros(80, dtype=torch.int64)
        for i in np.arange(80):
            if i in self.known_indices_coco:
                label_to_index[i] = torch.where(self.known_indices_coco==i)[0]
            else:
                label_to_index[i] = 2000
        return label_to_index

    def init_weights(self):
        super(YOLOXHeadModified, self).init_weights()
        # Use prior in model initialization to improve stability
        bias_init = bias_init_with_prob(0.01)
        for conv_cls, conv_obj in zip(self.multi_level_conv_cls,
                                      self.multi_level_conv_obj):
            #conv_cls.bias.data.fill_(bias_init)
            conv_obj.bias.data.fill_(bias_init)

    def forward_single(self, x, cls_convs, reg_convs, conv_cls, conv_reg,
                       conv_obj):
        """Forward feature of a single scale level."""

        cls_feat = cls_convs(x)
        reg_feat = reg_convs(x)

        cls_score = conv_cls(cls_feat)
        bbox_pred = conv_reg(reg_feat)
        objectness = conv_obj(reg_feat)

        return cls_score, bbox_pred, objectness

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """

        return multi_apply(self.forward_single, feats,
                           self.multi_level_cls_convs,
                           self.multi_level_reg_convs,
                           self.multi_level_conv_cls,
                           self.multi_level_conv_reg,
                           self.multi_level_conv_obj)

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   objectnesses,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network outputs of a batch into bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * embedding_dim, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.  Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, do nms before return boxes.
                Default True.
        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        # this method is only called during testing

        #  logit scaling is clipped to 100
        if self.logit_scale > 100:
            self.logit_scale = 100
        
        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
        cfg = self.test_cfg if cfg is None else cfg
        scale_factors = [img_meta['scale_factor'] for img_meta in img_metas]

        num_imgs = len(img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1) #.sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        flatten_priors = torch.cat(mlvl_priors)

        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        if rescale:
            flatten_bboxes[..., :4] /= flatten_bboxes.new_tensor(
                scale_factors).unsqueeze(1)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_scores = flatten_cls_scores[img_id]
            logits = self._get_cosine_sim_coco(cls_scores) * self.logit_scale
            
            # compute probabilities (softmax) for known and unknown classes separately
            if self.use_zsd:
                # model should only predict unknown classes
                cls_scores_known = torch.zeros_like(logits[:, self.known_indices_coco])
            else:
                cls_scores_known = F.softmax(logits[:, self.known_indices_coco], dim=1)

            if self.use_gzsd or self.use_zsd:
                cls_scores_unknown = F.softmax(logits[:, self.unknown_indices_coco], dim=1)
            else:
                # model should only predict known classes
                cls_scores_unknown = torch.zeros_like(logits[:, self.unknown_indices_coco])
            
            # reorder separate softmax scores to one matrix with 80 columns
            resort_indices = torch.argsort(torch.cat((self.known_indices_coco, self.unknown_indices_coco)))
            cls_scores = torch.cat((cls_scores_known, cls_scores_unknown), dim=1)[:, resort_indices]

            score_factor = flatten_objectness[img_id]
            bboxes = flatten_bboxes[img_id]

            result_list.append(
                self._bboxes_nms(cls_scores, bboxes, score_factor, cfg))

        return result_list

    def _bbox_decode(self, priors, bbox_preds):
        xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
        whs = bbox_preds[..., 2:].exp() * priors[:, 2:]

        tl_x = (xys[..., 0] - whs[..., 0] / 2)
        tl_y = (xys[..., 1] - whs[..., 1] / 2)
        br_x = (xys[..., 0] + whs[..., 0] / 2)
        br_y = (xys[..., 1] + whs[..., 1] / 2)

        decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
        return decoded_bboxes

    def _bboxes_nms(self, cls_scores, bboxes, score_factor, cfg):
        # this method is only called during testing
        # throw out all priors with VERY bad objectness (known and unknown classes alike)
        valid_mask_obj = score_factor > 0.1
        cls_scores = cls_scores[valid_mask_obj]
        bboxes = bboxes[valid_mask_obj]
        score_factor = score_factor[valid_mask_obj]

        if cls_scores.numel() == 0:
            return bboxes, torch.zeros_like(score_factor)

        # cls_scores for known classes are scaled by objectness
        cls_scores_known_scaled = cls_scores[:,self.known_indices_coco] * score_factor[:,None]
        cls_scores_unknown = cls_scores[:,self.unknown_indices_coco]
        
        # COMPUTATION FOR VALID MASK IF THRESHOLDS FOR SEEN AND UNSEEN CLASSES ARE NOT IDENTICAL
        #max_scores_known, _ = torch.max(cls_scores_known_scaled, 1)
        #max_scores_unknown, _ = torch.max(cls_scores_unknown, 1)
        #cls_unknown_thr = 0.45
        #valid_mask = torch.logical_or(max_scores_known >= cfg.score_thr, max_scores_unknown >= cls_unknown_thr)
        # attention: if you change the thresholds such that the thresholds for seen and unseen classes
        # are not identical, you have to set either cls_scores_unknown or cls_scores_known_scaled to zero
        # (depending on which threshold is greater) where threshold is not archieved
        # e.g. if threshold for unseen > threshold for seen:
        #cls_scores_unknown = torch.where(cls_scores_unknown >= cls_unknown_thr, cls_scores_unknown, torch.zeros_like(cls_scores_unknown))

        resort_indices = torch.argsort(torch.cat((self.known_indices_coco, self.unknown_indices_coco)))
        scores = torch.index_select(torch.cat((cls_scores_known_scaled, cls_scores_unknown), dim=1), 1, resort_indices)
        max_scores, labels = torch.max(scores, 1)
        valid_mask = max_scores >= cfg.score_thr

        bboxes = bboxes[valid_mask]
        scores = max_scores[valid_mask]
        labels = labels[valid_mask]

        if labels.numel() == 0:
            return bboxes, labels
        else:
            dets, keep = batched_nms(bboxes, scores, labels, cfg.nms)
            return dets, labels[keep]

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             objectnesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        """
        # unmodified yolox_head does not handle gt_bboxes_ignore either
        assert gt_bboxes_ignore is None

        # logit scaling is clipped to 100
        if self.logit_scale > 100:
            self.logit_scale = 100

        # split up input into coco and imagenet data
        num_imgs_total = len(img_metas)
        imagenet_mask = torch.zeros(num_imgs_total).bool().to('cuda')
        for i in range(len(gt_bboxes)):
            if gt_bboxes[i].numel() == 0 and gt_labels[i].numel() == 1:
                imagenet_mask[i] = 1
        coco_mask = ~imagenet_mask
        
        num_imgs_coco = torch.count_nonzero(coco_mask)
        num_imgs_imagenet = torch.count_nonzero(imagenet_mask)
        coco_ratio = num_imgs_coco/num_imgs_total
        imagenet_ratio = num_imgs_imagenet/num_imgs_total

        if num_imgs_coco:
            cls_scores_coco = [cls_score[coco_mask] for cls_score in cls_scores]
            bbox_preds_coco = [bbox_pred[coco_mask] for bbox_pred in bbox_preds]
            objectnesses_coco = [obj[coco_mask] for obj in objectnesses]
            gt_bboxes_coco = [gt_bboxes[i] for i in range(num_imgs_total) if coco_mask[i]]
            gt_labels_coco = [gt_labels[i] for i in range(num_imgs_total) if coco_mask[i]]
            img_metas_coco = [img_metas[i] for i in range(num_imgs_total) if coco_mask[i]]
   
            loss_dict_coco = self.loss_coco(cls_scores_coco, bbox_preds_coco, objectnesses_coco,
                gt_bboxes_coco, gt_labels_coco, img_metas_coco)
        else:
            loss_dict_coco = dict.fromkeys(['loss_cls','loss_bbox','loss_obj', 'loss_l1'], torch.tensor(0.).to('cuda'))

        if num_imgs_imagenet:
            cls_scores_imagenet = [cls_score[imagenet_mask] for cls_score in cls_scores]
            bbox_preds_imagenet = [bbox_pred[imagenet_mask] for bbox_pred in bbox_preds]
            objectnesses_imagenet = [obj[imagenet_mask] for obj in objectnesses]
            gt_bboxes_imagenet = [gt_bboxes[i] for i in range(num_imgs_total) if imagenet_mask[i]]
            gt_labels_imagenet = [gt_labels[i] for i in range(num_imgs_total) if imagenet_mask[i]]
            img_metas_imagenet = [img_metas[i] for i in range(num_imgs_total) if imagenet_mask[i]]

            loss_dict_imagenet = self.loss_imagenet(cls_scores_imagenet, bbox_preds_imagenet, objectnesses_imagenet,
                gt_bboxes_imagenet, gt_labels_imagenet, img_metas_imagenet)
        else:
            loss_dict_imagenet = dict.fromkeys(['loss_cls','loss_bbox','loss_obj', 'loss_l1'], torch.tensor(0.).to('cuda'))

        # recombine and scale losses (obj and bbox losses only depend on COCO data)
        loss_cls = coco_ratio * loss_dict_coco['loss_cls'] + imagenet_ratio * loss_dict_imagenet['loss_cls']
        loss_bbox = coco_ratio * loss_dict_coco['loss_bbox']
        # add imagenet obj loss to backpropagate 0 on imagenet data
        loss_obj = coco_ratio * loss_dict_coco['loss_obj'] + imagenet_ratio * loss_dict_imagenet['loss_obj']

        loss_dict = dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_obj=loss_obj)

        if self.use_l1:
            # bbox l1 loss only depend on COCO data
            loss_l1 = loss_dict_coco['loss_l1']
            loss_dict.update(loss_l1=loss_l1)

        return loss_dict

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses'))
    def loss_imagenet(self,
             cls_scores,
             bbox_preds,
             objectnesses,
             gt_bboxes,
             gt_labels,
             img_metas):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        """
        num_imgs = len(img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.cls_out_channels)
            for cls_pred in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        (cls_pos_masks, obj_pos_masks, bbox_pos_masks, cls_targets, obj_targets, bbox_targets,
         l1_targets, num_fg_imgs, ious) = multi_apply(
             self._get_target_single_imagenet, flatten_cls_preds.detach(),
             flatten_objectness.detach(),
             flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
             flatten_bboxes.detach(), gt_bboxes, gt_labels)

        num_total_samples = max(sum(num_fg_imgs), 1)

        cls_pos_masks = torch.cat(cls_pos_masks, 0)
        obj_pos_masks = torch.cat(obj_pos_masks, 0)
        bbox_pos_masks = torch.cat(bbox_pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        ious = torch.cat(ious, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        # use these lines if we change loss calculation such that there exists gradients
        # currently the following lines return 0 without gradients
        #loss_bbox = self.loss_bbox(
        #    flatten_bboxes.view(-1, 4)[bbox_pos_masks],
        #    bbox_targets) / num_total_samples

        loss_bbox = torch.tensor(0.).to('cuda')
        loss_obj = self.loss_obj(flatten_objectness.view(-1, 1)[obj_pos_masks],
                                 obj_targets) / num_total_samples

        # use the next line, if we want to backpropagate objectness loss different from 0
        # (ZSD_YOLO backpropagate obj loss = 0.3)
        # loss_obj = self.loss_obj(flatten_objectness.view(-1, 1), obj_targets) / num_total_samples

        loss_cls = self.loss_cls(
            self._get_cosine_sim_imagenet(flatten_cls_preds.view(-1, self.cls_out_channels)[cls_pos_masks]),
            self.logit_scale,
            cls_targets.long(),
            ious) / num_total_samples

        loss_dict = dict(
            loss_cls=loss_cls, loss_bbox=loss_bbox, loss_obj=loss_obj)

        if self.use_l1:
            loss_l1 = self.loss_l1(
                flatten_bbox_preds.view(-1, 4)[bbox_pos_masks],
                l1_targets) / num_total_samples
            loss_dict.update(loss_l1=loss_l1)

        return loss_dict

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses'))
    def loss_coco(self,
             cls_scores,
             bbox_preds,
             objectnesses,
             gt_bboxes,
             gt_labels,
             img_metas):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        """

        num_imgs = len(img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.cls_out_channels)
            for cls_pred in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        (cls_pos_masks, obj_pos_masks, bbox_pos_masks, cls_targets_seen, obj_targets, bbox_targets,
         l1_targets, num_fg_imgs, ious) = multi_apply(
             self._get_target_single_coco, flatten_cls_preds.detach(),
             flatten_objectness.detach(),
             flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
             flatten_bboxes.detach(), gt_bboxes, gt_labels)

        num_total_samples = max(sum(num_fg_imgs), 1)

        cls_pos_masks = torch.cat(cls_pos_masks, 0)
        obj_pos_masks = torch.cat(obj_pos_masks, 0)
        bbox_pos_masks = torch.cat(bbox_pos_masks, 0)
        cls_targets_seen = torch.cat(cls_targets_seen, 0)
        obj_targets = torch.cat(obj_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        ious = torch.cat(ious, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        loss_bbox = self.loss_bbox(
            flatten_bboxes.view(-1, 4)[bbox_pos_masks],
            bbox_targets) / num_total_samples
        loss_obj = self.loss_obj(flatten_objectness.view(-1, 1)[obj_pos_masks],
                                 obj_targets) / num_total_samples

        cosine_sim = self._get_cosine_sim_coco(flatten_cls_preds.view(-1, self.cls_out_channels)[cls_pos_masks])
        cosine_sim_seen = torch.index_select(cosine_sim, 1, self.known_indices_coco)

        loss_cls = self.loss_cls(
            cosine_sim_seen,
            self.logit_scale,
            cls_targets_seen.long(),
            ious) / num_total_samples

        loss_dict = dict(
            loss_cls=loss_cls, loss_bbox=loss_bbox, loss_obj=loss_obj)

        if self.use_l1:
            loss_l1 = self.loss_l1(
                flatten_bbox_preds.view(-1, 4)[bbox_pos_masks],
                l1_targets) / num_total_samples
            loss_dict.update(loss_l1=loss_l1)

        return loss_dict

    @torch.no_grad()
    def _get_target_single_imagenet(self, cls_preds, objectness, priors, decoded_bboxes,
                           gt_bboxes, gt_labels):
        """On COCO data: Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, embedding_dim].
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors].
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """
        num_gts = gt_labels.size(0)
        assert num_gts == 1

        # objectness and bbox loss should be calculated nowhere 
        obj_pos_mask = torch.zeros_like(objectness).to(torch.bool)
        bbox_pos_mask = torch.zeros_like(objectness).to(torch.bool)
        
        # choose the prior with the highest classification value for the gt_label
        # note that this does not require softmax
        pred = self._get_cosine_sim_imagenet(cls_preds)

        # we factor the objectness into the foreground_mask, so that we do not positively classify noise
        # the cutoff-value is equal to the cutoff-value in _bboxes_nms()
        # this is another part that probably only makes sense with COCO pre-training
        preds_for_gt = pred[:, gt_labels]
        obj_scores = objectness.unsqueeze(1).sigmoid()
        preds_with_obj = torch.where(obj_scores > 0.1, preds_for_gt, torch.zeros_like(preds_for_gt))

        cls_pos_mask = torch.zeros_like(objectness).to(torch.bool)
        if len(torch.nonzero(preds_with_obj)) > 0:
            if isinstance(self.loss_cls, HingeEmbeddingLoss):
                cls_target = F.one_hot(gt_labels,1000) * 2 - 1
            else:
                cls_target = gt_labels
            pos_ind = torch.argmax(preds_with_obj)
            cls_pos_mask[pos_ind] = 1
            num_pos_per_img = 1

            # for Imagenet data ious can not be calculated since we don't have bounding boxes
            # we assume iuo = 0.3 (see YOLO9000 paper)
            pos_ious = self.pos_ious
        else:
            # No target
            if isinstance(self.loss_cls, HingeEmbeddingLoss):
                cls_target = cls_preds.new_zeros((0, 1000))
            else:
                cls_target = cls_preds.new_zeros((0))

            num_pos_per_img = 0
            pos_ious = cls_preds.new_zeros((0))

        # setting the targets to [] means setting the loss to zero
        # if we wanted to learn an objectness from the imagenet data, this would be the place to set it
        bbox_target = cls_preds.new_zeros((0, 4))
        l1_target = cls_preds.new_zeros((0, 4))
        obj_target = cls_preds.new_zeros((0, 1))
        # use the next lines, if we want to propagate obj_loss for imagenet data
        #obj_target = cls_preds.new_zeros((num_priors, 1))
        #obj_target[pos_ind] = 1

        return (cls_pos_mask, obj_pos_mask, bbox_pos_mask, cls_target, obj_target,
                bbox_target, l1_target, num_pos_per_img, pos_ious)

    @torch.no_grad()
    def _get_target_single_coco(self, cls_preds, objectness, priors, decoded_bboxes,
                           gt_bboxes, gt_labels):
        """On COCO data: Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, embedding_dim].
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors].
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """

        num_priors = priors.size(0)
        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)

        # objectness loss should be calculated everywhere 
        obj_pos_mask = torch.ones_like(objectness).to(torch.bool)

        # No target
        if num_gts == 0:
            if isinstance(self.loss_cls, HingeEmbeddingLoss):
                cls_target_seen = cls_preds.new_zeros((0, torch.numel(self.known_indices_coco)))
            else:
                cls_target_seen = cls_preds.new_zeros((0))
            bbox_target = cls_preds.new_zeros((0, 4))
            l1_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_priors, 1))
            cls_pos_mask = cls_preds.new_zeros(num_priors).bool()
            bbox_pos_mask = cls_preds.new_zeros(num_priors).bool()
            pos_ious = cls_preds.new_zeros((0))
            return (cls_pos_mask, obj_pos_mask, bbox_pos_mask, cls_target_seen, obj_target,
                    bbox_target, l1_target, 0, pos_ious)

        # YOLOX uses center priors with 0.5 offset to assign targets,
        # but use center priors without offset to regress bboxes.
        offset_priors = torch.cat(
            [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)

        assign_result = self.assigner.assign(
            cls_preds, #.sigmoid() * objectness.unsqueeze(1).sigmoid(),
            objectness,
            offset_priors,
            decoded_bboxes,
            gt_bboxes,
            gt_labels,
            self.label_embeddings_coco,
            self.logit_scale)

        sampling_result = self.sampler.sample(assign_result, priors, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        num_pos_per_img = pos_inds.size(0)

        pos_ious = assign_result.max_overlaps[pos_inds]

        if isinstance(self.loss_cls, HingeEmbeddingLoss):
            cls_target = F.one_hot(sampling_result.pos_gt_labels,
                                self.num_classes) * 2 - 1
            cls_target_seen = torch.index_select(cls_target, 1, self.known_indices_coco)
        else:
            cls_target = sampling_result.pos_gt_labels
            cls_target_seen = self.label_id_to_seen_index[cls_target].long()

        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1
        bbox_target = sampling_result.pos_gt_bboxes
        l1_target = cls_preds.new_zeros((num_pos_per_img, 4))
        if self.use_l1:
            l1_target = self._get_l1_target(l1_target, bbox_target,
                                            priors[pos_inds])
        cls_pos_mask = torch.zeros_like(objectness).to(torch.bool)
        cls_pos_mask[pos_inds] = 1
        bbox_pos_mask = torch.zeros_like(objectness).to(torch.bool)
        bbox_pos_mask[pos_inds] = 1

        return (cls_pos_mask, obj_pos_mask, bbox_pos_mask, cls_target_seen, obj_target,
                bbox_target, l1_target, num_pos_per_img, pos_ious)

    def _get_l1_target(self, l1_target, gt_bboxes, priors, eps=1e-8):
        """Convert gt bboxes to center offset and log width height."""
        gt_cxcywh = bbox_xyxy_to_cxcywh(gt_bboxes)
        l1_target[:, :2] = (gt_cxcywh[:, :2] - priors[:, :2]) / priors[:, 2:]
        l1_target[:, 2:] = torch.log(gt_cxcywh[:, 2:] / priors[:, 2:] + eps)
        return l1_target
    
    def _get_cosine_sim_coco(self, cls_output):
        """Calculates scaled cosine similarity for all COCO labels"""
        cls_output = cls_output  / torch.norm(cls_output , dim=1)[:, None]
        cosine_sim = cls_output @ self.label_embeddings_coco
        return cosine_sim

    def _get_cosine_sim_imagenet(self, cls_output):
        """Calculates scaled cosine similarity for all Imagenet labels"""
        cls_output = cls_output  / torch.norm(cls_output , dim=1)[:, None]
        cosine_sim = cls_output @ self.label_embeddings_imagenet
        return cosine_sim