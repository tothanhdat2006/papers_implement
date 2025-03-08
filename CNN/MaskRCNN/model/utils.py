import math

import torch
import torch.nn as nn

class Matcher:
    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, iou):
        """
        Arguments:
            iou (Tensor[M, N]): containing the pairwise quality between 
            M ground-truth boxes and N predicted boxes.

        Returns:
            label (Tensor[N]): positive (1) or negative (0) label for each predicted box,
            -1 means ignoring this box.
            matched_idx (Tensor[N]): indices of gt box matched by each predicted box.
        """
        value, matched_idx = iou.max(dim=0) # pred: (N, ), (N, )
        label = torch.full((iou.shape[1],), fill_value=-1, dtype=torch.float, device=iou.device) # (N, )

        label[value >= self.high_threshold] = 1
        label[value < self.low_threshold] = 0

        if self.allow_low_quality_matches:
            highest_quailty = iou.max(dim=1)[0] # gt: (M, )
            gt_pred_pairs = torch.where(iou == highest_quailty[:, None])[1]
            label[gt_pred_pairs] = 1

        return label, matched_idx

class BalancedPositiveNegativeSampler:
    def __init__(self, num_samples, positive_fraction):
        self.num_samples = num_samples
        self.positive_fraction = positive_fraction

    def __call__(self, label):
        positive = torch.where(label == 1)[0]
        negative = torch.where(label == 0)[0]

        num_pos = int(self.num_samples * self.positive_fraction)
        num_pos = min(positive.numel(), num_pos)
        num_neg = self.num_samples - num_pos
        num_neg = min(negative.numel(), num_neg)

        pos_perm = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
        neg_perm = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

        pos_idx = positive[pos_perm]
        neg_idx = negative[neg_perm]

        return pos_idx, neg_idx

class AnchorGenerator(nn.Module):
    def __init__(self, sizes, ratios):
        super().__init__()
        self.sizes = sizes
        self.ratios = ratios

        self.cell_anchor = None
        self._cache = {}

    def set_cell_anchor(self, dtype, device):
        if self.cell_anchor is not None:
            return
    
        sizes = torch.tensor(self.sizes, dtype=dtype, device=device)
        ratios = torch.tensor(self.ratios, dtype=dtype, device=device)

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios
        
        # Convert to matrix to perform outer product then reshape to 1D
        hs = (sizes[:, None] * h_ratios[None, :]).view(-1) # (N, )
        ws = (sizes[:, None] * w_ratios[None, :]).view(-1) # (N, )
        
        self.cell_anchor = torch.stack([-ws, -hs, ws, hs], dim=1) / 2 # (N, 4)
    
    def grid_anchor(self, grid_size, stride):
        dtype, device = self.cell_anchor.dtype, self.cell_anchor.device
        shift_x = torch.arange(0, grid_size[1], dtype=dtype, device=device) * stride[0]
        shift_y = torch.arange(0, grid_size[0], dtype=dtype, device=device) * stride[1]

        y, x = torch.meshgrid(shift_y, shift_x)
        x = x.reshape(-1)
        y = y.reshape(-1)
        shift = torch.stack((x, y, x, y), dim=1).reshape(-1, 1, 4)
        
        anchor = (shift + self.cell_anchor).reshape(-1, 4)
        return anchor
    
    def cached_grid_anchor(self, grid_size, stride):
        key = grid_size + stride
        if key in self._cache:
            return self._cache[key]
        anchor = self.grid_anchor(grid_size, stride)
        
        if len(self._cache) >= 3:
            self._cache.clear()
        self._cache[key] = anchor
        return anchor
    
    def __call__(self, feature, image_size):
        dtype, device = feature.dtype, feature.device
        grid_size = tuple(feature.shape[-2:])
        stride = tuple(int(i / g) for i, g in zip(image_size, grid_size))

        self.set_cell_anchor(dtype, device)
        
        anchor = self.cached_grid_anchor(grid_size, stride)
        return anchor

class BoxCoder:
    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_box, proposal):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor[N, 4]): reference boxes
            proposals (Tensor[N, 4]): boxes to be encoded
        """

        width = proposal[:, 2] - proposal[:, 0]
        height = proposal[:, 3] - proposal[:, 1]
        center_x = proposal[:, 0] + 0.5 * width
        center_y = proposal[:, 1] + 0.5 * height

        gt_width = reference_box[:, 2] - reference_box[:, 0]
        gt_height = reference_box[:, 3] - reference_box[:, 1]
        gt_center_x = reference_box[:, 0] + 0.5 * gt_width
        gt_center_y = reference_box[:, 1] + 0.5 * gt_height

        # Section 3.1.1 in Faster R-CNN paper
        # The k proposal are parameterized relative to k reference boxes
        # Section 3.1.2 in Faster R-CNN paper
        # tx = (x - x_a) / w_a  
        # ty = (y - y_a) / h_a
        # tw = log(w / w_a)
        # th = log(h / h_a)
        tx = self.weights[0] * (gt_center_x - center_x) / width 
        ty = self.weights[1] * (gt_center_y - center_y) / height
        tw = self.weights[2] * torch.log(gt_width / width)
        th = self.weights[3] * torch.log(gt_height / height)

        delta = torch.stack((tx, ty, tw, th), dim=1)
        return delta

    def decode(self, delta, box):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.
        
        Arguments:
            delta (Tensor[N, 4]): encoded boxes.
            boxes (Tensor[N, 4]): reference boxes.
        """

        # Reverse of the encode formula
        tx = delta[:, 0] / self.weights[0]
        ty = delta[:, 1] / self.weights[1]
        tw = delta[:, 2] / self.weights[2]
        th = delta[:, 3] / self.weights[3]

        tw = torch.clamp(tw, max=self.bbox_xform_clip)
        th = torch.clamp(th, max=self.bbox_xform_clip)

        width = box[:, 2] - box[:, 0]
        height = box[:, 3] - box[:, 1]
        center_x = box[:, 0] + 0.5 * width
        center_y = box[:, 1] + 0.5 * height

        pred_center_x = tx * width + center_x
        pred_center_y = ty * height + center_y
        pred_width = torch.exp(tw) * width
        pred_height = torch.exp(th) * height

        pred_box = torch.stack((pred_center_x - 0.5 * pred_width, 
                                pred_center_y - 0.5 * pred_height, 
                                pred_center_x + 0.5 * pred_width, 
                                pred_center_y + 0.5 * pred_height), dim=1)
        return pred_box
    
def box_iou(box_a, box_b):
    """
    Arguments:
        box_a (Tensor[N, 4])
        box_b (Tensor[M, 4])
    
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in box_a and box_b
    """

    lt = torch.max(box_a[:, None, :2], box_b[:, :2])
    rb = torch.min(box_a[:, None, 2:], box_b[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = torch.prod(box_a[:, 2:] - box_a[:, :2], dim=1)
    area_b = torch.prod(box_b[:, 2:] - box_b[:, :2], dim=1)

    iou = inter / (area_a[:, None] + area_b - inter)
    return iou

def process_box(box, score, image_shape, min_size):
    """
    Clip boxes to image shape and remove small boxes
    which are smaller than min_size

    Arguments:
        box (Tensor[N, 4]): boxes to be processed
        score (Tensor[N]): scores of each box
        image_shape (tuple): shape of the image
        min_size (int): minimum size of the box
    """

    box[:, [0, 2]] = box[:, [0, 2]].clamp(0, image_shape[1])
    box[:, [1, 3]] = box[:, [1, 3]].clamp(0, image_shape[0])

    w, h = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1] # calculate width and height of tensor of box's coordinates
    keep = torch.where((w >= min_size) & (h >= min_size))[0] # Tensor of indices where the the coordinates of the box is greater than min_size
    return box[keep], score[keep]

def nms(box, score, threshold):
    """
    Arguments:
        box (Tensor[N, 4])
        score (Tensor[N]): scores of the boxes.

    Returns:
        indices (Tensor): indices of the elements that are kept
    """
    return torch.ops.torchvision.nms(box, score, threshold)
