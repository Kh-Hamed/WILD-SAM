import torch.nn as nn
import torch
import numpy as np

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lambda_=1.0):
        ctx.lambda_ = lambda_
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None
    
class grlLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super(grlLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalLayer.apply(x, self.lambda_)

class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.grl = grlLayer(1.0)
        self.bc = self.build_adv()
        self.DIF_ext = self.build_DIF_extractor()

    def build_adv(self):
        layers = []
        input_channels = 32
        output_channels_list = [16, 8, 4, 1]

        for output_channels in output_channels_list:
            layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(output_channels, eps=1e-3, momentum=0.01))
            if output_channels != 1:  # No activation on the final layer
                layers.append(nn.ReLU())
            input_channels = output_channels

        return nn.Sequential(*layers)


    def build_DIF_extractor(self):
        layers = []
        input_channels = 256
        output_channels_list = [128, 64, 32]

        for output_channels in output_channels_list:
            layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(output_channels, eps=1e-3, momentum=0.01))
            layers.append(nn.ReLU())
            input_channels = output_channels

        return nn.Sequential(*layers)

    def process_gt_boxes(self, batch_dict, H, W):
        feature_map_size = [H, W]
        all_names = np.array(['bg', 'Vehicle', 'Pedestrian'])
        cur_class_names = np.array(['Vehicle', 'Pedestrian'])
        bs = batch_dict['batch_size']
        gt_boxes = batch_dict['gt_boxes']
        inds = gt_boxes.new_zeros((bs, 1, H, W)).bool()
        for bs_idx in range(bs):
            cur_gt_boxes = gt_boxes[bs_idx]
            gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

            gt_boxes_T = []

            for idx, name in enumerate(gt_class_names):
                if name not in cur_class_names:
                    continue
                temp_box = cur_gt_boxes[idx]
                temp_box[-1] = cur_class_names.tolist().index(name) + 1
                gt_boxes_T.append(temp_box[None, :])

            if len(gt_boxes_T) == 0:
                gt_boxes_T = cur_gt_boxes[:0, :]
            else:
                gt_boxes_T = torch.cat(gt_boxes_T, dim=0)

            x, y, z = gt_boxes_T[:, 0], gt_boxes_T[:, 1], gt_boxes_T[:, 2]
            coord_x = ((x - (-75.2)) / 0.10) / 8
            coord_y = ((y - (-75.2)) / 0.10) / 8
            coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  
            coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  
            center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
            center_int = center.int()
            offsets = torch.tensor([
                [i, j] for i in range(-2, 3) for j in range(-2, 3)
            ], device=center.device)
            center_expanded = center_int.unsqueeze(1).repeat(1, offsets.shape[0], 1)
            neighbor_indices = center_expanded + offsets  # Shape will be (N, 9, 2)
            neighbor_indices = neighbor_indices.reshape(-1, 2)
            valid_indices = (neighbor_indices[:, 0] >= 0) & (neighbor_indices[:, 0] < feature_map_size[0]) & \
                            (neighbor_indices[:, 1] >= 0) & (neighbor_indices[:, 1] < feature_map_size[1])
            neighbor_indices_valid = neighbor_indices[valid_indices].long().clone()
            inds[bs_idx, 0, neighbor_indices_valid[:, 0], neighbor_indices_valid[:, 1]] = True
        
        batch_dict['bc_mask_unbalanced'] = 1.0 * inds
        
        return batch_dict

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
        #################################################################################################
        batch_dict = self.process_gt_boxes(batch_dict, H, W)
        spatial_features_DI = self.DIF_ext(spatial_features.clone().detach())
        batch_dict['spatial_features'] = torch.cat((spatial_features, spatial_features_DI), dim=1)
        #################################################################################################
        # batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        if self.training:
            x = self.grl(spatial_features_DI)
            logits = self.bc(x)
            bs = batch_dict['batch_size']
            bs_s = batch_dict['bs_s']
            bs_t = bs - bs_s
            zeros_tensor = torch.zeros(bs_t, 1, H, W)  # Shape: (B/2, 1, H, W)
            ones_tensor = torch.ones(bs_t, 1, H, W)    # Shape: (B/2, 1, H, W)
            lables = torch.cat([zeros_tensor, ones_tensor], dim=0)  # Shape: (B, 1, H, W)
            indices_s = torch.randperm(bs_s)[:bs_t]
            indices_t = torch.arange(bs_s, bs_s + bs_t)
            indexs = torch.cat((indices_s, indices_t)).to(logits.device)
            batch_dict['bc_pred'] = logits[indexs].sigmoid()
            batch_dict['bc_label'] = lables.to(logits.device)
            batch_dict['bc_mask'] = batch_dict['bc_mask_unbalanced'][indexs]
        return batch_dict
