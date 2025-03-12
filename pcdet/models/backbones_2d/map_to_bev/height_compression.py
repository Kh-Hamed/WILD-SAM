import torch.nn as nn
import torch
import numpy as np

# class GradientReversalLayer(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, lambda_=1.0):
#         ctx.lambda_ = lambda_
#         return input

#     @staticmethod
#     def backward(ctx, grad_output):
#         return -ctx.lambda_ * grad_output, None
    
# class grlLayer(nn.Module):
#     def __init__(self, lambda_=1.0):
#         super(grlLayer, self).__init__()
#         self.lambda_ = lambda_

#     def forward(self, x):
#         return GradientReversalLayer.apply(x, self.lambda_)

class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        # self.grl = grlLayer(1.0)
        # self.dc = self.build_adv()
        # self.task_head = self.build_adv(task=True)
        # self.DIF_ext = self.build_DIF_extractor()

    # def build_adv(self, task = False):
    #     layers = []
    #     input_channels = 32
    #     output_channels_list = [32, 16, 16, 8, 8, 4, 4]

    #     for output_channels in output_channels_list:
    #         layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=True))
    #         layers.append(nn.BatchNorm2d(output_channels, eps=1e-3, momentum=0.01))
    #         layers.append(nn.ReLU())

    #         input_channels = output_channels

    #     layers.append(nn.Conv2d(input_channels, 1, kernel_size=3, stride=1, padding=1, bias=True))
    #     fc = nn.Sequential(*layers)
    #     if isinstance(fc[-1], nn.Conv2d)and task:
    #         fc[-1].bias.data.fill_(-2.19)

    #     return fc


    # def build_DIF_extractor(self):
    #     layers = []
    #     input_channels = 256
    #     output_channels_list = [256, 128, 128, 64, 64, 32, 32]

    #     for output_channels in output_channels_list:
    #         layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False))
    #         layers.append(nn.BatchNorm2d(output_channels, eps=1e-3, momentum=0.01))
    #         layers.append(nn.ReLU())
    #         input_channels = output_channels

    #     return nn.Sequential(*layers)

    # def process_gt_boxes(self, batch_dict, H, W):
    #     feature_map_size = [H, W]
    #     all_names = np.array(['bg', 'Vehicle', 'Pedestrian'])
    #     cur_class_names = np.array(['Vehicle', 'Pedestrian'])
    #     bs = batch_dict['batch_size']
    #     gt_boxes = batch_dict['gt_boxes']
    #     inds = gt_boxes.new_zeros((bs, 1, H, W)).bool()
    #     for bs_idx in range(bs):
    #         cur_gt_boxes = gt_boxes[bs_idx]
    #         gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

    #         gt_boxes_T = []

    #         for idx, name in enumerate(gt_class_names):
    #             if name not in cur_class_names:
    #                 continue
    #             temp_box = cur_gt_boxes[idx]
    #             temp_box[-1] = cur_class_names.tolist().index(name) + 1
    #             gt_boxes_T.append(temp_box[None, :])

    #         if len(gt_boxes_T) == 0:
    #             gt_boxes_T = cur_gt_boxes[:0, :]
    #         else:
    #             gt_boxes_T = torch.cat(gt_boxes_T, dim=0)

    #         x, y, z = gt_boxes_T[:, 0], gt_boxes_T[:, 1], gt_boxes_T[:, 2]
    #         coord_x = ((x - (-75.2)) / 0.10) / 8
    #         coord_y = ((y - (-75.2)) / 0.10) / 8
    #         coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  
    #         coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  
    #         center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
    #         center_int = center.int()
    #         offsets = torch.tensor([
    #             [i, j] for i in range(-2, 3) for j in range(-2, 3)
    #         ], device=center.device)
    #         center_expanded = center_int.unsqueeze(1).repeat(1, offsets.shape[0], 1)
    #         neighbor_indices = center_expanded + offsets  # Shape will be (N, 9, 2)
    #         neighbor_indices = neighbor_indices.reshape(-1, 2)
    #         valid_indices = (neighbor_indices[:, 0] >= 0) & (neighbor_indices[:, 0] < feature_map_size[0]) & \
    #                         (neighbor_indices[:, 1] >= 0) & (neighbor_indices[:, 1] < feature_map_size[1])
    #         neighbor_indices_valid = neighbor_indices[valid_indices].long().clone()
    #         inds[bs_idx, 0, neighbor_indices_valid[:, 1], neighbor_indices_valid[:, 0]] = True
    #     # size = feature_map_size[0]
    #     # center = size // 2
    #     # sigma = size // 6
    #     # x = torch.linspace(0, size - 1, size)
    #     # y = torch.linspace(0, size - 1, size)
    #     # x, y = torch.meshgrid(x, y)
    #     # gaussian_weight = (1.0 - torch.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))).to(inds.device).unsqueeze(0).unsqueeze(0)
        
    #     batch_dict['dc_mask_fg'] = 1.0 * inds
        
    #     return batch_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        ################################################################################################
        # spatial_features_DI = self.DIF_ext(spatial_features.clone().detach())
        # batch_dict['spatial_features'] = torch.cat((spatial_features, spatial_features_DI), dim=1)
        # batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        ################################################################################################
        # if self.training:
        #     batch_dict = self.process_gt_boxes(batch_dict, H, W)
        #     x = self.grl(spatial_features_DI)
        #     logits = self.dc(x)
        #     task_pred = self.task_head(x)
        #     bs = batch_dict['batch_size']
        #     width = W   # Define width
        #     height = H  # Define height

        #     center_x = width // 2
        #     center_y = height // 2
        #     radius = torch.min(torch.tensor([width, height])) // 4  # Use torch.min

        #     # Create coordinate arrays
        #     x = torch.linspace(0, width - 1, width)
        #     y = torch.linspace(0, height - 1, height)
        #     x, y = torch.meshgrid(x, y, indexing='ij')  # Ensure correct indexing

        #     # Compute the binary mask: 1 inside the circle, 0 outside
        #     circle_mask = ((x - center_x) ** 2 + (y - center_y) ** 2) >= radius ** 2
        #     circle_mask = circle_mask.float()
        #     lables = circle_mask.repeat(bs, 1, 1).unsqueeze(1).to(logits.device)

        #     batch_dict['task_pred'] = task_pred.sigmoid()
        #     batch_dict['dc_pred'] = logits.sigmoid()
        #     batch_dict['dc_label'] = lables.to(logits.device)
        return batch_dict
