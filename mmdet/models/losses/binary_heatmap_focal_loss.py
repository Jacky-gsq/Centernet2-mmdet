import mmcv
import torch.nn as nn
import torch

from ..builder import LOSSES
from .utils import weighted_loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def binary_heatmap_focal_loss(
    pred,
    targets,
    pos_inds,
    alpha: float = -1,
    beta: float = 4,
    gamma: float = 2,
    sigmoid_clamp: float = 1e-4,
    ignore_high_fp: float = -1.,
):
    """
    Args:
        inputs:  (sum_l N*Hl*Wl,)
        targets: (sum_l N*Hl*Wl,)
        pos_inds: N
    Returns:
        Loss tensor with the reduction option applied.
    """
    clamp_pred = torch.clamp(pred.sigmoid_(), min=sigmoid_clamp, max=1-sigmoid_clamp)
    neg_weights = torch.pow(1 - targets, beta)
    pos_pred = clamp_pred[pos_inds] # N
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, gamma)
    neg_loss = torch.log(1 - clamp_pred) * torch.pow(clamp_pred, gamma) * neg_weights
    if ignore_high_fp > 0:
        not_high_fp = (clamp_pred < ignore_high_fp).float()
        neg_loss = not_high_fp * neg_loss

    pos_loss = - pos_loss.sum()
    neg_loss = - neg_loss.sum()

    if alpha >= 0:
        pos_loss = alpha * pos_loss
        neg_loss = (1 - alpha) * neg_loss

    return pos_loss, neg_loss


@LOSSES.register_module()
class BinaryHeatmapFocalLoss(nn.Module):
    """GaussianFocalLoss is a variant of focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.

    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negative samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 alpha=0.25,
                 beta=4,
                 gamma=2,
                 sigmoid_clamp=0.001,
                 ignore_high_fp=0.85,
                 reduction='none',
                 loss_weight=1.0):
        super(BinaryHeatmapFocalLoss, self).__init__()
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.sigmoid_clamp=sigmoid_clamp
        self.ignore_high_fp=ignore_high_fp
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                pos_inds,
                # weight=None,
                # avg_factor=None,
                # reduction_override=None
                ):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction
                in gaussian distribution.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        # assert reduction_override in (None, 'none', 'mean', 'sum')
        # reduction = (
        #     reduction_override if reduction_override else self.reduction)
        weighted_pos_loss, weighted_neg_loss = self.loss_weight * binary_heatmap_focal_loss(
                pred=pred,
                target=target,
                pos_inds=pos_inds,
                alpha=self.alpha, 
                beta=self.beta, 
                gamma=self.gamma,
                sigmoid_clamp=self.sigmoid_clamp,
                ignore_high_fp=self.ignore_high_fp,)
        return weighted_pos_loss, weighted_neg_loss
