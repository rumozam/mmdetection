# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


def hinge_max_loss(pred,
                   label,
                   ious,
                   weight=None,
                   reduction='mean',
                   avg_factor=None,
                   top_k=1,
                   margin=1):
    """Calculate HingeMaxLoss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C),
            C is the number of classes.
        label (torch.Tensor): The learning label of the prediction
            with shape (N).
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        top_k (int, optional): The number of maximal terms contributing to
            the loss. Defaults to 1, in which case the top_k operation behaves
            like a max.
        margin (float, optional): The margin.
    Returns:
        torch.Tensor: The calculated loss
    """
    (N, C) = pred.shape

    if torch.numel(label) == 0:
        return 0

    # predictions for correct classes
    # shape: (N)
    mask_correct = torch.zeros_like(pred).scatter_(1, label.unsqueeze(1), 1.).bool()
    correct_label_preds = pred[mask_correct]

    # predictions for wrong classes
    # shape: (N, C-1)
    wrong_label_preds = pred[~mask_correct].view(N, C-1)

    # margin + sim_wrong - sim_correct
    term = torch.full_like(wrong_label_preds, margin) + \
        wrong_label_preds - correct_label_preds.reshape(N,-1).expand(N, C-1)

    # max(0, term)
    input = torch.where(term > 0, term, torch.zeros_like(wrong_label_preds))

    # max(max(0, term))
    loss = torch.topk(input, k=top_k, dim=1, sorted=False)[0]

    loss = loss * ious.reshape(-1,1)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


@LOSSES.register_module()
class HingeMaxLoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 margin=1,
                 top_k=1):
        """HingeMaxLoss.

        Args:
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
            margin (float, optional): The margin.
            top_k (int, optional): The number of maximal terms contributing to
                the loss. Defaults to 1, in which case the top_k operation behaves
                like a max.

        """
        super(HingeMaxLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.margin = margin
        self.top_k = top_k

    def forward(self,
                cls_score,
                logit_scale,
                label,
                ious,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                margin=1,
                top_k=1):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            logit_scale: not needed.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss. Options are "none", "mean" and "sum".
            margin (float, optional): The margin.
        Returns:
            torch.Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_cls = self.loss_weight * hinge_max_loss(
            cls_score,
            label,
            ious,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            margin=self.margin,
            top_k=self.top_k)
        return loss_cls
