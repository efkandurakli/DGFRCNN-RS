from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss, maskrcnn_loss, maskrcnn_inference


class RoIHeads(RoIHeads):
    def __init__(
        self,
        box_roi_pool,
        box_head,
        box_predictor,
        # Faster R-CNN training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh,
        nms_thresh,
        detections_per_img,
        # Mask
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
    ):
        super().__init__(
            box_roi_pool,
            box_head,
            box_predictor,
            fg_iou_thresh,
            bg_iou_thresh,
            batch_size_per_image,
            positive_fraction,
            bbox_reg_weights,
            score_thresh,
            nms_thresh,
            detections_per_img,
            mask_roi_pool,
            mask_head,
            mask_predictor
        )

    def forward(
            self,
            features,  # type: Dict[str, Tensor]
            proposals,  # type: List[Tensor]
            image_shapes,  # type: List[Tuple[int, int]]
            targets=None,  # type: Optional[List[Dict[str, Tensor]]]
        ):
            # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
            """
            Args:
                features (List[Tensor])
                proposals (List[Tensor[N, 4]])
                image_shapes (List[Tuple[H, W]])
                targets (List[Dict])
            """
            if targets is not None:
                for t in targets:
                    # TODO: https://github.com/pytorch/pytorch/issues/26731
                    floating_point_types = (torch.float, torch.double, torch.half)
                    if not t["boxes"].dtype in floating_point_types:
                        raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                    if not t["labels"].dtype == torch.int64:
                        raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")
                    if self.has_keypoint():
                        if not t["keypoints"].dtype == torch.float32:
                            raise TypeError(f"target keypoints must of float type, instead got {t['keypoints'].dtype}")

            box_domains = []
            if self.training:
                proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
                for i in range(len(targets)):
                    box_domain = targets[i]["domain"]
                    box_domain = box_domain.repeat(proposals[i].shape[0])
                    box_domains.append(box_domain)

                box_domains = torch.cat(box_domains, dim=0)
            else:
                labels = None
                regression_targets = None
                matched_idxs = None
            
            box_features = self.box_roi_pool(features, proposals, image_shapes)
            box_features = self.box_head(box_features, box_domains, labels)
            class_logits, box_regression = self.box_predictor(box_features)

            result: List[Dict[str, torch.Tensor]] = []
            losses = {}
            if self.training:
                if labels is None:
                    raise ValueError("labels cannot be None")
                if regression_targets is None:
                    raise ValueError("regression_targets cannot be None")
                loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
                losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
            else:
                boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
                num_images = len(boxes)
                for i in range(num_images):
                    result.append(
                        {
                            "boxes": boxes[i],
                            "labels": labels[i],
                            "scores": scores[i],
                        }
                    )

            if self.has_mask():
                mask_proposals = [p["boxes"] for p in result]
                mask_domains = []
                if self.training:
                    if matched_idxs is None:
                        raise ValueError("if in training, matched_idxs should not be None")

                    # during training, only focus on positive boxes
                    num_images = len(proposals)
                    mask_proposals = []
                    pos_matched_idxs = []
                    for img_id in range(num_images):
                        pos = torch.where(labels[img_id] > 0)[0]
                        mask_domain = targets[img_id]["domain"]
                        mask_domain = mask_domain.repeat(proposals[img_id].shape[0])
                        mask_domains.append(mask_domain[pos])
                        mask_proposals.append(proposals[img_id][pos])
                        pos_matched_idxs.append(matched_idxs[img_id][pos])
                    
                    mask_domains = torch.cat(mask_domains, dim=0)
                else:
                    pos_matched_idxs = None

                if self.mask_roi_pool is not None:
                    mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                    mask_features = self.mask_head(mask_features, mask_domains)
                    mask_logits = self.mask_predictor(mask_features)
                else:
                    raise Exception("Expected mask_roi_pool to be not None")

                loss_mask = {}
                if self.training:
                    if targets is None or pos_matched_idxs is None or mask_logits is None:
                        raise ValueError("targets, pos_matched_idxs, mask_logits cannot be None when training")

                    gt_masks = [t["masks"] for t in targets]
                    gt_labels = [t["labels"] for t in targets]
                    rcnn_loss_mask = maskrcnn_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)
                    loss_mask = {"loss_mask": rcnn_loss_mask}
                else:
                    labels = [r["labels"] for r in result]
                    masks_probs = maskrcnn_inference(mask_logits, labels)
                    for mask_prob, r in zip(masks_probs, result):
                        r["masks"] = mask_prob

                losses.update(loss_mask)

            return result, losses
