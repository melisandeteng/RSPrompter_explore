import copy
import warnings
import einops
import numpy as np
import torch
from mmcv.cnn import build_norm_layer, ConvModule
from mmcv.ops import point_sample
from mmengine import ConfigDict
from mmengine.dist import is_main_process
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from peft import get_peft_config, get_peft_model
from torch import nn, Tensor
from transformers import SamConfig
from transformers.models.sam.modeling_sam import SamVisionEncoder, SamMaskDecoder, SamPositionalEmbedding, \
    SamPromptEncoder, SamModel, SamVisionEncoderOutput
from typing import List, T, Tuple, Optional, Dict, Union
from mmdet.models import MaskRCNN, StandardRoIHead, FCNMaskHead, SinePositionalEncoding, Mask2Former, Mask2FormerHead, \
    MaskFormerFusionHead, BaseDetector
from mmdet.models.task_modules import SamplingResult
from mmdet.models.utils import unpack_gt_instances, empty_instances, multi_apply, \
    get_uncertain_point_coords_with_randomness
from mmdet.registry import MODELS
from mmdet.structures import SampleList, DetDataSample, OptSampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import OptConfigType, MultiConfig, ConfigType, InstanceList, reduce_mean
import torch.nn.functional as F

from mmpretrain.models import LayerNorm2d

    
@MODELS.register_module()
class RSPrompterAnchorDSM(MaskRCNN):
    def __init__(
            self,
            shared_image_embedding,
            decoder_freeze=True,
            *args,
            **kwargs):
        peft_config = kwargs.get('backbone', {}).get('peft_config', {})
        super().__init__(*args, **kwargs)
        self.shared_image_embedding = MODELS.build(shared_image_embedding)
        self.decoder_freeze = decoder_freeze

        self.frozen_modules = []
        if peft_config is None:
            self.frozen_modules += [self.backbone]
        if self.decoder_freeze:
            self.frozen_modules += [
                self.shared_image_embedding,
                self.roi_head.mask_head.mask_decoder,
                self.roi_head.mask_head.no_mask_embed,
            ]
        self._set_grad_false(self.frozen_modules)

    def _set_grad_false(self, module_list=[]):
        for module in module_list:
            module.eval()
            if isinstance(module, nn.Parameter):
                module.requires_grad = False
            for param in module.parameters():
                param.requires_grad = False

    def get_image_wide_positional_embeddings(self, size):
        target_device = self.shared_image_embedding.shared_image_embedding.positional_embedding.device
        target_dtype = self.shared_image_embedding.shared_image_embedding.positional_embedding.dtype
        grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size

        positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)  # channel x height x width

    def extract_feat(self, batch_inputs) -> Tuple[Tensor]:
        im_inputs, dsm_inputs =  batch_inputs
        vision_outputs = self.backbone(batch_inputs)
        
        dsm_outputs = self.backbone(dsm_inputs)
        if isinstance(vision_outputs, SamVisionEncoderOutput):
            image_embeddings = vision_outputs[0]
            vision_hidden_states = vision_outputs[1]
        elif isinstance(vision_outputs, tuple):
            image_embeddings = vision_outputs[0]
            vision_hidden_states = vision_outputs
        else:
            raise NotImplementedError
        if isinstance(dsm_outputs, SamVisionEncoderOutput):
            
            dsm_hidden_states = dsm_outputs[1]
        elif isinstance(vision_outputs, tuple):
            dsm_hidden_states = dsm_outputs
        else:
            raise NotImplementedError

        image_positional_embeddings = self.get_image_wide_positional_embeddings(size=image_embeddings.shape[-1])
        # repeat with batch size
        batch_size = image_embeddings.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

        x = self.neck(vision_hidden_states)
        dsm_x = self.neck(dsm_hidden_states)
        return x, dsm_x, image_embeddings, image_positional_embeddings

    def loss(self, batch_inputs,
             batch_data_samples: SampleList) -> dict:
    
        x, dsm_x,image_embeddings, image_positional_embeddings = self.extract_feat(batch_inputs)

        losses = dict()
        # RPN forward and loss
        proposal_cfg = self.train_cfg.get('rpn_proposal',
                                          self.test_cfg.rpn)
        rpn_data_samples = copy.deepcopy(batch_data_samples)
        # set cat_id of gt_labels to 0 in RPN
        for data_sample in rpn_data_samples:
            data_sample.gt_instances.labels = \
                torch.zeros_like(data_sample.gt_instances.labels)

        rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
            x, rpn_data_samples, proposal_cfg=proposal_cfg)
        # avoid get same name with roi_head loss
        keys = rpn_losses.keys()
        for key in list(keys):
            if 'loss' in key and 'rpn' not in key:
                rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
        losses.update(rpn_losses)

        roi_losses = self.roi_head.loss(
            x, rpn_results_list, batch_data_samples,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
        )
        losses.update(roi_losses)

        return losses

    def predict(self,
                batch_inputs,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        batch_inputs = im_inputs, dsm_inputs
        x, dsm_x, image_embeddings, image_positional_embeddings = self.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
        )
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

@MODELS.register_module()
class RSPrompterAnchorMaskHeadDSM(FCNMaskHead, BaseModule):
    def __init__(
            self,
            mask_decoder,
            in_channels,
            roi_feat_size=14,
            per_pointset_point=5,
            with_sincos=True,
            multimask_output=False,
            attention_similarity=None,
            target_embedding=None,
            output_attentions=None,
            class_agnostic=False,
            loss_mask: ConfigType = dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
            init_cfg=None,
            *args,
            **kwargs):
        BaseModule.__init__(self, init_cfg=init_cfg)

        self.in_channels = in_channels
        self.roi_feat_size = roi_feat_size
        self.per_pointset_point = per_pointset_point
        self.with_sincos = with_sincos
        self.multimask_output = multimask_output
        self.attention_similarity = attention_similarity
        self.target_embedding = target_embedding
        self.output_attentions = output_attentions

        self.mask_decoder = MODELS.build(mask_decoder)

        prompt_encoder = dict(
            type='RSSamPromptEncoder',
            hf_pretrain_name=copy.deepcopy(mask_decoder.get('hf_pretrain_name')),
            init_cfg=copy.deepcopy(mask_decoder.get('init_cfg')),
        )
        prompt_encoder = MODELS.build(prompt_encoder)
        prompt_encoder.init_weights()
        self.no_mask_embed = prompt_encoder.prompt_encoder.no_mask_embed

        if with_sincos:
            num_sincos = 2
        else:
            num_sincos = 1
        self.point_emb = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_channels*roi_feat_size**2//4, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels * num_sincos * per_pointset_point)
        )

        self.loss_mask = MODELS.build(loss_mask)
        self.class_agnostic = class_agnostic

    def init_weights(self) -> None:
        BaseModule.init_weights(self)

    def forward(self,
                x,
                dsm_x, 
                image_embeddings,
                image_positional_embeddings,
                roi_img_ids=None,
                ):
        img_bs = image_embeddings.shape[0]
        roi_bs = x.shape[0]
        image_embedding_size = image_embeddings.shape[-2:]

        point_embedings = self.point_emb(torch.cat(x, dsm_x))
        point_embedings = einops.rearrange(point_embedings, 'b (n c) -> b n c', n=self.per_pointset_point)
        if self.with_sincos:
            point_embedings = torch.sin(point_embedings[..., ::2]) + point_embedings[..., 1::2]

        # (B * N_set), N_point, C
        sparse_embeddings = point_embedings.unsqueeze(1)
        num_roi_per_image = torch.bincount(roi_img_ids.long())
        # deal with the case that there is no roi in an image
        num_roi_per_image = torch.cat([num_roi_per_image, torch.zeros(img_bs - len(num_roi_per_image), device=num_roi_per_image.device, dtype=num_roi_per_image.dtype)])

        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(roi_bs, -1, image_embedding_size[0], image_embedding_size[1])
        # get image embeddings with num_roi_per_image
        image_embeddings = image_embeddings.repeat_interleave(num_roi_per_image, dim=0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(num_roi_per_image, dim=0)

        low_res_masks, iou_predictions, mask_decoder_attentions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output,
            attention_similarity=self.attention_similarity,
            target_embedding=self.target_embedding,
            output_attentions=self.output_attentions,
        )
        h, w = low_res_masks.shape[-2:]
        low_res_masks = low_res_masks.reshape(roi_bs, -1, h, w)
        iou_predictions = iou_predictions.reshape(roi_bs, -1)
        return low_res_masks, iou_predictions

    def get_targets(self, sampling_results: List[SamplingResult],
                    batch_gt_instances: InstanceList,
                    rcnn_train_cfg: ConfigDict) -> Tensor:
        pos_proposals = [res.pos_priors for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        gt_masks = [res.masks for res in batch_gt_instances]
        mask_targets_list = []
        mask_size = rcnn_train_cfg.mask_size
        device = pos_proposals[0].device
        for pos_gt_inds, gt_mask in zip(pos_assigned_gt_inds, gt_masks):
            if len(pos_gt_inds) == 0:
                mask_targets = torch.zeros((0,) + mask_size, device=device, dtype=torch.float32)
            else:
                mask_targets = gt_mask[pos_gt_inds.cpu()].to_tensor(dtype=torch.float32, device=device)
            mask_targets_list.append(mask_targets)
        mask_targets = torch.cat(mask_targets_list)
        return mask_targets

    def loss_and_target(self, mask_preds: Tensor,
                        sampling_results: List[SamplingResult],
                        batch_gt_instances: InstanceList,
                        rcnn_train_cfg: ConfigDict) -> dict:
        mask_targets = self.get_targets(
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=rcnn_train_cfg)

        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        # resize to mask_targets size
        mask_preds = F.interpolate(mask_preds, size=mask_targets.shape[-2:], mode='bilinear', align_corners=False)

        loss = dict()
        if mask_preds.size(0) == 0:
            loss_mask = mask_preds.sum()
        else:
            if self.class_agnostic:
                loss_mask = self.loss_mask(mask_preds, mask_targets,
                                           torch.zeros_like(pos_labels))
            else:
                loss_mask = self.loss_mask(mask_preds, mask_targets,
                                           pos_labels)
        loss['loss_mask'] = loss_mask
        return dict(loss_mask=loss, mask_targets=mask_targets)

    def _predict_by_feat_single(self,
                                mask_preds: Tensor,
                                bboxes: Tensor,
                                labels: Tensor,
                                img_meta: dict,
                                rcnn_test_cfg: ConfigDict,
                                rescale: bool = False,
                                activate_map: bool = False) -> Tensor:
        scale_factor =np.ones((2,2))#bboxes.new_tensor(img_meta['scale_factor']).repeat((1, 2))
        img_h, img_w = img_meta['ori_shape'][:2]
        if not activate_map:
            mask_preds = mask_preds.sigmoid()
        else:
            # In AugTest, has been activated before
            mask_preds = bboxes.new_tensor(mask_preds)

        if rescale:  # in-placed rescale the bboxes
            bboxes = bboxes #/= scale_factor
        else:
            w_scale, h_scale = scale_factor[0, 0], scale_factor[0, 1]
            img_h = np.round(img_h * h_scale.item()).astype(np.int32)
            img_w = np.round(img_w * w_scale.item()).astype(np.int32)
        threshold = rcnn_test_cfg.mask_thr_binary
        im_mask = F.interpolate(mask_preds, size=img_meta['batch_input_shape'], mode='bilinear', align_corners=False).squeeze(1)

        scale_factor_w, scale_factor_h = (1,1) #img_meta['scale_factor']
        ori_rescaled_size = (img_h * scale_factor_h, img_w * scale_factor_w)
        im_mask = im_mask[:, :int(ori_rescaled_size[0]), :int(ori_rescaled_size[1])]

        h, w = img_meta['ori_shape']
        im_mask = F.interpolate(im_mask.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(1)

        if threshold >= 0:
            im_mask = im_mask >= threshold
        else:
            # for visualization and debugging
            im_mask = (im_mask * 255).to(dtype=torch.uint8)
        return im_mask