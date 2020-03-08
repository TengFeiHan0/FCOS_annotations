"""
This file contains specific functions for computing losses of FCOS
file
"""

import torch
from torch.nn import functional as F
from torch import nn
import os
from ..utils import concat_box_prediction_layers
from fcos_core.layers import IOULoss
from fcos_core.layers import SigmoidFocalLoss
from fcos_core.modeling.matcher import Matcher
from fcos_core.modeling.utils import cat
from fcos_core.structures.boxlist_ops import boxlist_iou
from fcos_core.structures.boxlist_ops import cat_boxlist


INF = 100000000


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor


class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.FCOS.LOSS_GAMMA,
            cfg.MODEL.FCOS.LOSS_ALPHA
        )
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.center_sampling_radius = cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS
        self.iou_loss_type = cfg.MODEL.FCOS.IOU_LOSS_TYPE
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS

        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss(self.iou_loss_type)
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        #self.cenerness_loss_func = nn.MSELoss(reduction='sum')

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1.0):
        '''
        This code is from
        https://github.com/yqyao/FCOS_PLUS/blob/0d20ba34ccc316650d8c30febb2eb40cb6eaae37/
        maskrcnn_benchmark/modeling/rpn/fcos/loss.py#L42
        '''
        num_gts = gt.shape[0]
        K = len(gt_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(
                xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0]
            )
            center_gt[beg:end, :, 1] = torch.where(
                ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1]
            )
            center_gt[beg:end, :, 2] = torch.where(
                xmax > gt[beg:end, :, 2],
                gt[beg:end, :, 2], xmax
            )
            center_gt[beg:end, :, 3] = torch.where(
                ymax > gt[beg:end, :, 3],
                gt[beg:end, :, 3], ymax
            )
            beg = end
        left = gt_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs[:, None]
        top = gt_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def prepare_targets(self, points, targets):
        """
        定义每一层需要关注的物体大小，变量object_sizes_of_interest；
        根据location的位置向量和bbox标注信息，计算每一个location的期望label和回归值，
        由self.compute_targets_for_locations()函数实现；
        将一个batch中不同图像的同一层的值（期望label和回归值）进行拼接，并返回

        """
        #?这个阈值怎么用呢？
        #首先，我们要遍历每一个fmap的所以位置，对于一个fmap上的(x,y) ,我们先计算出[t, b ,l ,r]，
        #然后取其中的最大值 max[t, b, l,r] ,然后确定这个m落是否落在了自己所对应的阈值范围内。
        #如果是，正例；否则，负例。
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ] # 每一层特征图对应的检测的box的比例大小
        # 创建5个tensor对应着5个特征图
        # 每个tensor的行数为对应的特征图的点的个数，列数为2
        # 例如第一个tensor的size为(len(points_per_level), 2)，元素为重复的[-1, 64]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )
        # expanded_object_sizes_of_interest维度
        # expanded_object_sizes_of_interest[0]为重复的[-1, 64]
        # expanded_object_sizes_of_interest[1]为重复的[64, 128]
        # [e.shape for e in expanded_object_sizes_of_interest]
        # [torch.Size([16128, 2]), torch.Size([4032, 2]), torch.Size([1008, 2]), torch.Size([252, 2]), torch.Size([66, 2])]

        # 在行的维度上concat
        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        #![16800, 4200, 1050, 273, 77]每张图片肯定都不一样
        self.num_points_per_level = num_points_per_level
        
        # 所有特征图点拼接，每个点代表其对应原图方框视野的中心点坐标
        points_all_level = torch.cat(points, dim=0)
        # 论文中分配box策略
        # 根据预设比例对不同level特征图分配不同的大小的box
        # labels：5层特征图每个point对应的label size：[torch.Size([21486])]
        # reg_targets：5层特征图每个point对应的box的四个坐标 size：[torch.Size([21486, 4])]
        #!(注，未经处理，有正有负)
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )

        for i in range(len(labels)):
            # 将label[0]切分为了[torch.Size([16128]), torch.Size([4032]), 
            #torch.Size([1008]), torch.Size([252]), torch.Size([66])]
            # 对应这5个特征图
            # reg_targets同理
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)
        labels_level_first = []
        reg_targets_level_first = []
        # 将多个图片的同一level的label和reg_targets合并成一个tensor
        for level in range(len(points)):
            # 对一个batch内的所有图片的所有point在行的维度上进行concat
            # 后追加到labels_level_first
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )
            #同理
            reg_targets_per_level = torch.cat([
                reg_targets_per_im[level]
                for reg_targets_per_im in reg_targets
            ], dim=0)

            if self.norm_reg_targets:
                reg_targets_per_level = reg_targets_per_level / self.fpn_strides[level]
            reg_targets_level_first.append(reg_targets_per_level)
        #labels特征图中每个点对应其在原图方框区域中心点的 label (背景为 0) 大小近似为 [ 5 , 所有特征图点数] 
        #reg_targets特征图中每个点其在原图方框区域的中心点所对应的框的四个值 大小近似为 [ 5 , 所有特征图点数 ， 4] 

        return labels_level_first, reg_targets_level_first

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox
            # 每个框对应的类别
            labels_per_im = targets_per_im.get_field("labels")
            #面积
            area = targets_per_im.area()
            #see figure1 in original paper 4D real vector t∗ = (l∗, t∗, r∗, b∗)
            # are the distances from the location to the four sides of the bounding box
            #!Eq1
            #l∗ = x − x(i)0 , t∗ = y − y(i)0
            #r∗ = x(i)1 − x, b∗ = y(i)1 − y
            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]

            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)
            #torch.Size([18134, 2, 4]) [5个特征图像素点数，当前图片框数，[l, t, r, b]
            if self.center_sampling_radius > 0:
                is_in_boxes = self.get_sample_region(
                    bboxes,
                    self.fpn_strides,
                    self.num_points_per_level,
                    xs, ys,
                    radius=self.center_sampling_radius
                )
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            #判断每个点是否在该层关注大小内
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])
            
            #torch.Size([所有层特征图点数, 真实标注框数]) # 在列这个维度重复每个框的面积
            locations_to_gt_area = area[None].repeat(len(locations), 1)
            # 将不在标注框内的点的框大小设置为 最大
            locations_to_gt_area[is_in_boxes == 0] = INF
            # 将每个点对应框大小超出该层设置的点对应的框的面积设置为 最大
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)
            # 每个点对对应的框的四个值 torch.Size([20267, 4])
            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            # 每个点对应每个框的 lable
            labels_per_im = labels_per_im[locations_to_gt_inds]
            # 设置不符合条件的框的 label 为 0
            labels_per_im[locations_to_min_area == INF] = 0

            labels.append(labels_per_im)# torch.Size([20267]) 每个点对应一个 label 背景为 0
            reg_targets.append(reg_targets_per_im)
            # labels = 特征图中每个点对应其在原图方框区域中心点的 label (背景为 0) 
            #reg_targets = 特征图中每个点其在原图方框区域的中心点所对应的框的四个值(注，未经处理，有正有负)
            #?为什么不在这里直接返回所有在bbox里即需要计算的点的 4 个值？？？ 可以将不符合的点的值设置为 INF?

        return labels, reg_targets

    def compute_centerness_targets(self, reg_targets):
        # see 论文中的eq3
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression, centerness, targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        """ 
        根据输入target，利用prepare_targets()函数计算位置点的类别和回归的期望输出，labels和reg_targets;
        根据变量维度将box_cls、box_regression、centerness、labels、reg_targets变量展成二维或一维向量，消除batch_size、特征层个数等维度；
        根据期望输出labels和实际输出box_cls计算分类损失；
        根据labels，查找前景物体位置序号；
        提取前景物体的box_regression与centerness，利用compute_centerness_targets()函数计算中心度的目标值；
        分别计算中心度与回归对应的损失，并返回
        """
        N = box_cls[0].size(0) # batch 照片数
        num_classes = box_cls[0].size(1)
        # labels = 特征图中每个点对应其在原图方框区域中心点的 label (背景为 0) 大小为 [ 5层 , 所有特征图点数]
        # reg_targets = 特征图中每个点其在原图方框区域的中心点所对应的框的四个值 大小为 [ 5层 , 所有特征图点数 ，4]
        labels, reg_targets = self.prepare_targets(locations, targets)

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        for l in range(len(labels)):
            ## 每一层 先 变成 [batch_size,h,w,80] 再 reshpe -> (-1,80) ---> 每个点的预测类别
            #下面的同理
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            centerness_flatten.append(centerness[l].reshape(-1))
        # 在行的维度上进行增加
        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)
        # 提取有类别的特征图中的点
        # 正例点的索引（有 label 的点的索引）
        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        # pos_inds : tensor([ 7971,  7972,  7973,  8123,  8124,  8125,  8275,  8276,  8277, 17133,
        # 17134, 17135, 20057, 20058, 20059, 20068, 20069, 20070, 20076, 20077,
        # 20078, 20087, 20088, 20089, 20095, 20096, 20097, 20106, 20107, 20108,
        # 20243, 20244], device='cuda:0')
        
        # 根据pos_inds提取正样本
        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        num_gpus = get_num_gpus()
        # sync num_pos from all gpus
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)
        # 分类loss用的 focal loss
        #分类结果和真实的标签分类结果计算loss
        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten.int()
        ) / num_pos_avg_per_gpu

        if pos_inds.numel() > 0:
            #计算真实的centerness
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)

            # average sum_centerness_targets from all gpus,
            # which is used to normalize centerness-weighed reg loss
            sum_centerness_targets_avg_per_gpu = \
                reduce_sum(centerness_targets.sum()).item() / float(num_gpus)
            
            # 计算正例预测的框与真实框的 IOU loss
            #!这里的中心度作为权重输入进去
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            ) / sum_centerness_targets_avg_per_gpu
            #BCE loss
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            ) / num_pos_avg_per_gpu
        else:
            reg_loss = box_regression_flatten.sum()
            reduce_sum(centerness_flatten.new_tensor([0.0]))
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss


def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator
