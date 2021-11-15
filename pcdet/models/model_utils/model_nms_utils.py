import torch

from ...ops.iou3d_nms import iou3d_nms_utils


def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):
    src_box_scores = box_scores
    if score_thresh is not None:
        scores_mask = (box_scores >= score_thresh)
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]

    selected = []
    if box_scores.shape[0] > 0:
        box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
        boxes_for_nms = box_preds[indices]
        keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
        )
        selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]
    return selected, src_box_scores[selected]


def multi_thresh(box_scores, box_labels, box_preds, nms_config, score_thresh=None):
    src_box_scores = box_scores
    selected = []
    selected_end = []
    if score_thresh is not None:
        for i, cur_thresh in enumerate(score_thresh):
            mask = ((i+1) == box_labels)
            cur_box_scores = box_scores[mask]
            cur_box_preds = box_preds[mask]
            score_mask = (cur_box_scores >= cur_thresh)
            cur_box_scores = cur_box_scores[score_mask]
            cur_box_preds = cur_box_preds[score_mask]

            if cur_box_scores.shape[0] > 0:
                cur_box_scores_nms, indices = torch.topk(cur_box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, cur_box_scores.shape[0]))
                cur_box_for_nms = cur_box_preds[indices]
                keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                    cur_box_for_nms, cur_box_scores_nms, nms_config.NMS_THRESH, **nms_config
                )
                cur_selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

                score_idxs = score_mask.nonzero().view(-1)
                original_idxs = mask.nonzero().view(-1)
                cur_selected = score_idxs[cur_selected]
                cur_selected = original_idxs[cur_selected]
                selected.append(cur_selected)
    if len(selected):
        selected = torch.cat(selected, dim=0)
        box_for_nms_end = box_preds[selected]
        box_scores_nms_end = box_scores[selected]

        keep_idx_end, selected_scores_end = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
            box_for_nms_end, box_scores_nms_end, nms_config.NMS_THRESH, **nms_config
        )
        selected_end = selected[keep_idx_end]
    return selected_end, src_box_scores[selected_end]


def multi_thresh_one_nms(box_scores, box_labels, box_preds, nms_config, score_thresh=None):
    src_box_scores = box_scores
    selected = []
    if score_thresh is not None:
        box_scores_nms = []
        indices = []
        for i, cur_thresh in enumerate(score_thresh):
            mask = ((i+1) == box_labels)
            cur_box_scores = box_scores[mask]
            cur_box_preds = box_preds[mask]
            score_mask = (cur_box_scores >= cur_thresh)
            cur_box_scores = cur_box_scores[score_mask]
            cur_box_preds = cur_box_preds[score_mask]

            if cur_box_scores.shape[0] > 0:
                cur_box_scores_nms, cur_indices = torch.topk(cur_box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, cur_box_scores.shape[0]))
                box_scores_nms.extend(cur_box_scores_nms)
                indices.extend(cur_indices)

        boxes_for_nms = box_preds[indices]
        keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
            boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
        )
        selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

        if score_thresh is not None:
            original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]
    if len(selected):
        selected = torch.cat(selected, dim=0)
    return selected, src_box_scores[selected]

def multi_classes_nms(cls_scores, box_preds, nms_config, score_thresh=None):
    """
    Args:
        cls_scores: (N, num_class)
        box_preds: (N, 7 + C)
        nms_config:
        score_thresh:

    Returns:

    """
    pred_scores, pred_labels, pred_boxes = [], [], []
    for k in range(cls_scores.shape[1]):
        if score_thresh is not None:
            scores_mask = (cls_scores[:, k] >= score_thresh)
            box_scores = cls_scores[scores_mask, k]
            cur_box_preds = box_preds[scores_mask]
        else:
            box_scores = cls_scores[:, k]
            cur_box_preds = box_preds

        selected = []
        if box_scores.shape[0] > 0:
            box_scores_nms, indices = torch.topk(box_scores, k=min((nms_config.NMS_PRE_MAXSIZE)[k], box_scores.shape[0]))
            boxes_for_nms = cur_box_preds[indices]
            keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                    boxes_for_nms[:, 0:7], box_scores_nms, (nms_config.NMS_THRESH)[k], **nms_config
            )
            selected = indices[keep_idx[:(nms_config.NMS_POST_MAXSIZE)[k]]]

        pred_scores.append(box_scores[selected])
        pred_labels.append(box_scores.new_ones(len(selected)).long() * k)
        pred_boxes.append(cur_box_preds[selected])

    pred_scores = torch.cat(pred_scores, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    pred_boxes = torch.cat(pred_boxes, dim=0)

    return pred_scores, pred_labels, pred_boxes


def multi_classes_bev_selected(cls_scores, box_preds, nms_config, score_thresh=None):
    """
    Args:
        cls_scores: (N, num_class)
        box_preds: (N, 7 + C)
        nms_config:
        score_thresh:

    Returns:

    """
    pred_scores, pred_labels, pred_boxes = [], [], []
    for k in range(cls_scores.shape[1]):
        if score_thresh is not None:
            scores_mask = (cls_scores[:, k] >= score_thresh)
            box_scores = cls_scores[scores_mask, k]
            cur_box_preds = box_preds[scores_mask]
        else:
            box_scores = cls_scores[:, k]
            cur_box_preds = box_preds

        if box_scores.shape[0] > 0:
            box_scores_nms, indices = torch.topk(box_scores, k=min((nms_config.NMS_PRE_MAXSIZE)[k], box_scores.shape[0]))

        pred_scores.append(box_scores[indices])
        pred_labels.append(box_scores.new_ones(len(indices)).long() * k)
        pred_boxes.append(cur_box_preds[indices])

    pred_scores = torch.cat(pred_scores, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    pred_boxes = torch.cat(pred_boxes, dim=0)

    return pred_scores, pred_labels, pred_boxes


def class_agnostic_nms_v2(box_scores, box_labels, box_preds, nms_config, score_thresh=None):
    src_box_scores = box_scores
    selected_idx = []
    for i in range(0, 3):
        mask = (box_labels == i)
        box_scores_mask = box_scores[mask]
        box_preds_mask = box_preds[mask]
        if(box_scores.shape[0] > 0):
            box_scores_nms, indices = torch.topk(box_scores_mask, k=min((nms_config.NMS_PRE_MAXSIZE)[i], box_scores_mask.shape[0]))
            boxes_for_nms = box_preds_mask[indices]
            keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                    boxes_for_nms[:, 0:7], box_scores_nms, (nms_config.NMS_THRESH)[i], **nms_config
            )
            selected = indices[keep_idx[:(nms_config.NMS_POST_MAXSIZE)[i]]]
        original_idxs = mask.nonzero().view(-1)
        selected = original_idxs[selected]
        selected_idx.append(selected)
    selected_idx = torch.cat(selected_idx, dim=-1)
    return selected_idx, src_box_scores[selected_idx]