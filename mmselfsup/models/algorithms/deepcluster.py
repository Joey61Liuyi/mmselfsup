# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.data import LabelData

from mmselfsup.core import SelfSupDataSample
from mmselfsup.registry import MODELS
from .base import BaseModel


@MODELS.register_module()
class DeepCluster(BaseModel):
    """DeepCluster.

    Implementation of `Deep Clustering for Unsupervised Learning
    of Visual Features <https://arxiv.org/abs/1807.05520>`_.
    The clustering operation is in `core/hooks/deepcluster_hook.py`.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors.
        head (dict): Config dict for module of head functions.
        data_preprocessor (dict or nn.Module, optional): Config to preprocess
            images. Defaults to None.
        init_cfg (dict or List[dict], optional): Config dict for weight
            initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 pretrained: Optional[str] = None,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            pretrained=pretrained,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        # re-weight
        self.num_classes = self.head.num_classes
        self.loss_weight = torch.ones((self.num_classes, ),
                                      dtype=torch.float32).cuda()
        self.loss_weight /= self.loss_weight.sum()

    def extract_feat(self, batch_inputs: List[torch.Tensor],
                     **kwarg) -> Tuple[torch.Tensor]:
        """Function to extract features from backbone.

        Args:
            batch_inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Tuple[torch.Tensor]: backbone outputs.
        """
        x = self.backbone(batch_inputs[0])
        return x

    def loss(self, batch_inputs: List[torch.Tensor],
             data_samples: List[SelfSupDataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            batch_inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        pseudo_label = torch.cat([
            data_sample.pseudo_label.clustering_label
            for data_sample in data_samples
        ])
        x = self.extract_feat(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        self.head.loss.class_weight = self.loss_weight
        loss = self.head(x, pseudo_label)
        losses = dict(loss=loss)
        return losses

    def predict(self, batch_inputs: List[torch.Tensor],
                data_samples: List[SelfSupDataSample],
                **kwargs) -> List[SelfSupDataSample]:
        """The forward function in testing.

        Args:
            batch_inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            List[SelfSupDataSample]: The prediction from model.
        """
        x = self.extract_feat(batch_inputs)  # tuple
        if self.with_neck:
            x = self.neck(x)
        outs = self.head.logits(x)
        keys = [f'head{i}' for i in self.backbone.out_indices]

        for i in range(len(outs)):
            prediction_data = {key: out for key, out in zip(keys, outs)}
            prediction = LabelData(**prediction_data)
            data_samples[i].pred_label = prediction
        return data_samples
