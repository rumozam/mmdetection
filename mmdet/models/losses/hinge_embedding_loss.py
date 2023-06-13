# Copyright (c) OpenMMLab. All rights reserved.
import torch
import math
from torch._C import AggregationType
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
import sys

from ..builder import LOSSES
from .utils import weight_reduce_loss


def hinge_embedding_loss(pred,
                  target,
                  margin,
                  ious,
                  weight=None,
                  reduction='mean',
                  avg_factor=None):
    """Calculate pytorch HingeEmbeddingLoss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C),
            C is the dimension of the predicted embeddings.
        label (torch.Tensor): The learning label of the prediction
            with shape (N).
        weight (torch.Tensor, optional): Sample-wise loss weight.
        red (str, optional): The method used to reduce the loss.

    Returns:
        torch.Tensor: The calculated loss
    """
    l1 = nn.L1Loss(reduction='none')
    x = l1(pred, target)
    loss = F.hinge_embedding_loss(
        x,
        target,
        margin,
        # can not use "sum" here because of iou scaling
        reduction="none"
    )
    loss = torch.sum(loss,1)

    # scale with ious
    loss = loss * ious

    # do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


@LOSSES.register_module()
class HingeEmbeddingLoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 margin=1.0):
        """pytorch HingeEmbeddingLoss

        Args:
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(HingeEmbeddingLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.margin = margin


    def forward(self,
                cls_score,
                logit_scale,
                label,
                ious,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
uction
        Args:
            cls_score (torch.Tensor): The prediction.
            logit_scale(torch.Tensor): Not needed for hinge loss.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss. Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_cls = self.loss_weight * hinge_embedding_loss(
            cls_score,
            label,
            self.margin,
            ious,
            weight,
            reduction,
            avg_factor)
        return loss_cls
