# -*- coding: utf-8 -*-
# CellViT networks and adaptions, without sharing encoders
#
# UNETR paper and code: https://github.com/tamasino52/UNETR
# SAM paper and code: https://segment-anything.com/
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Literal, Tuple, Union
import timm
from peft import LoraConfig, get_peft_model
from segment_anything.modeling.transformer import TwoWayTransformer

import numpy as np
import torch
import torch.nn as nn

from cell_segmentation.utils.post_proc_cellvit import DetectionCellPostProcessor

from .utils import Conv2DBlock, Deconv2DBlock, ViTCellViT, ViTCellViTDeit

from models.mlp.archs import DWConv, shiftedBlock, shiftmlp

from segment_anything import sam_model_registry

import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class DinoV3BackboneWrapper(nn.Module):
    """
    DINOv3 Backbone 包装器
    功能：
    1. 加载 timm 模型
    2. 支持 PEFT/LoRA 注入
    3. 前向传播时同时返回 [空间特征图] 和 [CLS token]
    """
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        # dynamic_img_size=True 允许输入不同分辨率，timm会自动插值位置编码
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True 
        )
        self.patch_size = self.model.patch_embed.patch_size[0]
        self.embed_dim = self.model.embed_dim
        self.is_peft = False

    def apply_peft(self, rank=16):
        """
        对 Backbone 应用 LoRA
        """
        
        print(f"正在为 Backbone 应用 LoRA (Rank={rank})...")
        # 针对 ViT 结构的常用 LoRA 配置
        # target_modules 包括 Attention 的 qkv, proj 和 MLP 的 fc1, fc2
        config = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            target_modules=["qkv", "proj", "fc1", "fc2"], 
            lora_dropout=0.05,
            bias="none",
            modules_to_save=[], # 如果需要训练 LayerNorm，可添加 "norm"
        )
        
        # 将 timm 模型包装为 PeftModel
        self.model = get_peft_model(self.model, config)
        self.is_peft = True
        self.model.print_trainable_parameters()
        
    def forward(self, x):
        """
        输入: [B, 3, H, W]
        输出: 
           - spatial_features: [B, Embed_Dim, H/P, W/P]
           - cls_token: [B, Embed_Dim]
        """
        B, C, H, W = x.shape
        
        # 1. 获取序列特征 [B, N_tokens, Dim]
        # 处理 PEFT 包装器：PeftModel 默认不暴露 forward_features，需要访问底层模型
        model_obj = self.model
        if self.is_peft:
            # PeftModel -> BaseTuner -> Original Model (timm)
            # 通常路径是 model.base_model.model
            if hasattr(model_obj, 'base_model') and hasattr(model_obj.base_model, 'model'):
                model_obj = model_obj.base_model.model
            elif hasattr(model_obj, 'base_model'):
                model_obj = model_obj.base_model
        
        # 确保我们调用的是 forward_features (跳过 Head，直接获取 tokens)
        if hasattr(model_obj, 'forward_features'):
            x_feats = model_obj.forward_features(x)
        else:
            # Fallback (不太可能发生，除非 timm 版本差异)
            x_feats = model_obj(x)

        # 2. 分离 CLS Token 和 Patch Tokens
        # timm 的 num_prefix_tokens 通常为 1 (CLS token)
        # 注意：如果是 PeftModel，属性可能需要通过 getattr 获取，但上面的 model_obj 已解包
        n_prefix = getattr(model_obj, 'num_prefix_tokens', 1)
        
        if n_prefix > 0:
            cls_token = x_feats[:, 0:n_prefix, :].mean(dim=1) # 获取 CLS token
            patch_tokens = x_feats[:, n_prefix:, :]           # 获取图像 patch 特征
        else:
            cls_token = x_feats.mean(dim=1) # 如果没有CLS token，用GAP代替
            patch_tokens = x_feats

        # 3. Reshape 恢复空间维度
        h_grid = H // self.patch_size
        w_grid = W // self.patch_size
        
        # [B, N, Dim] -> [B, H, W, Dim]
        # 此时 patch_tokens 已经是包含 LoRA 权重计算后的特征
        patch_tokens = patch_tokens.reshape(B, h_grid, w_grid, self.embed_dim)
        
        # [B, H, W, Dim] -> [B, Dim, H, W] 以符合 Conv2d 输入
        spatial_features = patch_tokens.permute(0, 3, 1, 2).contiguous()
        
        return spatial_features, cls_token

class CellViT(nn.Module):
    """CellViT Modell for cell segmentation. U-Net like network with vision transformer as backbone encoder

    Skip connections are shared between branches, but each network has a distinct encoder

    The modell is having multiple branches:
        * tissue_types: Tissue prediction based on global class token
        * nuclei_binary_map: Binary nuclei prediction
        * hv_map: HV-prediction to separate isolated instances
        * nuclei_type_map: Nuclei instance-prediction
        * [Optional, if regression loss]:
        * regression_map: Regression map for binary prediction

    Args:
        num_nuclei_classes (int): Number of nuclei classes (including background)
        num_tissue_classes (int): Number of tissue classes
        embed_dim (int): Embedding dimension of backbone ViT
        input_channels (int): Number of input channels
        depth (int): Depth of the backbone ViT
        num_heads (int): Number of heads of the backbone ViT
        extract_layers: (List[int]): List of Transformer Blocks whose outputs should be returned in addition to the tokens. First blocks starts with 1, and maximum is N=depth.
            Is used for skip connections. At least 4 skip connections needs to be returned.
        mlp_ratio (float, optional): MLP ratio for hidden MLP dimension of backbone ViT. Defaults to 4.
        qkv_bias (bool, optional): If bias should be used for query (q), key (k), and value (v) in backbone ViT. Defaults to True.
        drop_rate (float, optional): Dropout in MLP. Defaults to 0.
        attn_drop_rate (float, optional): Dropout for attention layer in backbone ViT. Defaults to 0.
        drop_path_rate (float, optional): Dropout for skip connection . Defaults to 0.
        regression_loss (bool, optional): Use regressive loss for predicting vector components.
            Adds two additional channels to the binary decoder, but returns it as own entry in dict. Defaults to False.
    """

    def __init__(
        self,
        num_nuclei_classes: int,
        num_tissue_classes: int,
        embed_dim: int,
        input_channels: int,
        depth: int,
        num_heads: int,
        extract_layers: List,
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
        regression_loss: bool = False,
    ):
        # For simplicity, we will assume that extract layers must have a length of 4
        super().__init__()
        assert len(extract_layers) == 4, "Please provide 4 layers for skip connections"

        self.patch_size = 16
        self.num_tissue_classes = num_tissue_classes
        self.num_nuclei_classes = num_nuclei_classes
        self.embed_dim = embed_dim
        self.input_channels = input_channels
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.extract_layers = extract_layers
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate

        self.encoder = ViTCellViT(
            patch_size=self.patch_size,
            num_classes=self.num_tissue_classes,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            extract_layers=self.extract_layers,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

        if self.embed_dim < 512:
            self.skip_dim_11 = 256
            self.skip_dim_12 = 128
            self.bottleneck_dim = 312
        else:
            self.skip_dim_11 = 512
            self.skip_dim_12 = 256
            self.bottleneck_dim = 512

        # version with shared skip_connections
        self.decoder0 = nn.Sequential(
            Conv2DBlock(3, 32, 3, dropout=self.drop_rate),
            Conv2DBlock(32, 64, 3, dropout=self.drop_rate),
        )  # skip connection after positional encoding, shape should be H, W, 64
        self.decoder1 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_12, 128, dropout=self.drop_rate),
        )  # skip connection 1
        self.decoder2 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, 256, dropout=self.drop_rate),
        )  # skip connection 2
        self.decoder3 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate)
        )  # skip connection 3

        self.regression_loss = regression_loss
        offset_branches = 0
        if self.regression_loss:
            offset_branches = 2
        self.branches_output = {
            "nuclei_binary_map": 2 + offset_branches,
            "hv_map": 2,
            "nuclei_type_maps": self.num_nuclei_classes,
        }

        self.nuclei_binary_map_decoder = self.create_upsampling_branch(
            2 + offset_branches
        )  # todo: adapt for helper loss
        self.hv_map_decoder = self.create_upsampling_branch(
            2
        )  # todo: adapt for helper loss
        self.nuclei_type_maps_decoder = self.create_upsampling_branch(
            self.num_nuclei_classes
        )

    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False) -> dict:
        """Forward pass

        Args:
            x (torch.Tensor): Images in BCHW style
            retrieve_tokens (bool, optional): If tokens of ViT should be returned as well. Defaults to False.

        Returns:
            dict: Output for all branches:
                * tissue_types: Raw tissue type prediction. Shape: (B, num_tissue_classes)
                * nuclei_binary_map: Raw binary cell segmentation predictions. Shape: (B, 2, H, W)
                * hv_map: Binary HV Map predictions. Shape: (B, 2, H, W)
                * nuclei_type_map: Raw binary nuclei type preditcions. Shape: (B, num_nuclei_classes, H, W)
                * [Optional, if retrieve tokens]: tokens
                * [Optional, if regression loss]:
                * regression_map: Regression map for binary prediction. Shape: (B, 2, H, W)
        """
        assert (
            x.shape[-2] % self.patch_size == 0
        ), "Img must have a shape of that is divisible by patch_size (token_size)"
        assert (
            x.shape[-1] % self.patch_size == 0
        ), "Img must have a shape of that is divisible by patch_size (token_size)"

        out_dict = {}

        classifier_logits, _, z = self.encoder(x)
        out_dict["tissue_types"] = classifier_logits

        z0, z1, z2, z3, z4 = x, *z

        # performing reshape for the convolutional layers and upsampling (restore spatial dimension)
        patch_dim = [int(d / self.patch_size) for d in [x.shape[-2], x.shape[-1]]]
        z4 = z4[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z3 = z3[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z2 = z2[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z1 = z1[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)

        if self.regression_loss:
            nb_map = self._forward_upsample(
                z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder
            )
            out_dict["nuclei_binary_map"] = nb_map[:, :2, :, :]
            out_dict["regression_map"] = nb_map[:, 2:, :, :]
        else:
            out_dict["nuclei_binary_map"] = self._forward_upsample(
                z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder
            )
        out_dict["hv_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.hv_map_decoder
        )
        out_dict["nuclei_type_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder
        )
        if retrieve_tokens:
            out_dict["tokens"] = z4

        return out_dict

    def _forward_upsample(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
        z4: torch.Tensor,
        branch_decoder: nn.Sequential,
    ) -> torch.Tensor:
        """Forward upsample branch

        Args:
            z0 (torch.Tensor): Highest skip
            z1 (torch.Tensor): 1. Skip
            z2 (torch.Tensor): 2. Skip
            z3 (torch.Tensor): 3. Skip
            z4 (torch.Tensor): Bottleneck
            branch_decoder (nn.Sequential): Branch decoder network

        Returns:
            torch.Tensor: Branch Output
        """
        b4 = branch_decoder.bottleneck_upsampler(z4)
        b3 = self.decoder3(z3)
        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        b2 = self.decoder2(z2)
        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))
        b1 = self.decoder1(z1)
        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        b0 = self.decoder0(z0)
        branch_output = branch_decoder.decoder0_header(torch.cat([b0, b1], dim=1))

        return branch_output

    def create_upsampling_branch(self, num_classes: int) -> nn.Module:
        """Create Upsampling branch

        Args:
            num_classes (int): Number of output classes

        Returns:
            nn.Module: Upsampling path
        """
        bottleneck_upsampler = nn.ConvTranspose2d(
            in_channels=self.embed_dim,
            out_channels=self.bottleneck_dim,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )
        decoder3_upsampler = nn.Sequential(
            Conv2DBlock(
                self.bottleneck_dim * 2, self.bottleneck_dim, dropout=self.drop_rate
            ),
            Conv2DBlock(
                self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate
            ),
            Conv2DBlock(
                self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate
            ),
            nn.ConvTranspose2d(
                in_channels=self.bottleneck_dim,
                out_channels=256,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder2_upsampler = nn.Sequential(
            Conv2DBlock(256 * 2, 256, dropout=self.drop_rate),
            Conv2DBlock(256, 256, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder1_upsampler = nn.Sequential(
            Conv2DBlock(128 * 2, 128, dropout=self.drop_rate),
            Conv2DBlock(128, 128, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder0_header = nn.Sequential(
            Conv2DBlock(64 * 2, 64, dropout=self.drop_rate),
            Conv2DBlock(64, 64, dropout=self.drop_rate),
            nn.Conv2d(
                in_channels=64,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        decoder = nn.Sequential(
            OrderedDict(
                [
                    ("bottleneck_upsampler", bottleneck_upsampler),
                    ("decoder3_upsampler", decoder3_upsampler),
                    ("decoder2_upsampler", decoder2_upsampler),
                    ("decoder1_upsampler", decoder1_upsampler),
                    ("decoder0_header", decoder0_header),
                ]
            )
        )

        return decoder

    def calculate_instance_map(
        self, predictions: OrderedDict, magnification: Literal[20, 40] = 40
    ) -> Tuple[torch.Tensor, List[dict]]:
        """Calculate Instance Map from network predictions (after Softmax output)

        Args:
            predictions (dict): Dictionary with the following required keys:
                * nuclei_binary_map: Binary Nucleus Predictions. Shape: (B, 2, H, W)
                * nuclei_type_map: Type prediction of nuclei. Shape: (B, self.num_nuclei_classes, H, W)
                * hv_map: Horizontal-Vertical nuclei mapping. Shape: (B, 2, H, W)
            magnification (Literal[20, 40], optional): Which magnification the data has. Defaults to 40.

        Returns:
            Tuple[torch.Tensor, List[dict]]:
                * torch.Tensor: Instance map. Each Instance has own integer. Shape: (B, H, W)
                * List of dictionaries. Each List entry is one image. Each dict contains another dict for each detected nucleus.
                    For each nucleus, the following information are returned: "bbox", "centroid", "contour", "type_prob", "type"
        """
        # reshape to B, H, W, C
        predictions_ = predictions.copy()
        predictions_["nuclei_type_map"] = predictions_["nuclei_type_map"].permute(
            0, 2, 3, 1
        )
        predictions_["nuclei_binary_map"] = predictions_["nuclei_binary_map"].permute(
            0, 2, 3, 1
        )
        predictions_["hv_map"] = predictions_["hv_map"].permute(0, 2, 3, 1)

        cell_post_processor = DetectionCellPostProcessor(
            nr_types=self.num_nuclei_classes, magnification=magnification, gt=False
        )
        instance_preds = []
        type_preds = []

        # for i in range(predictions_["nuclei_binary_map"].shape[0]):
        #     # 分别检查每个变量
        #     nuclei_type_tensor = torch.argmax(predictions_["nuclei_type_map"], dim=-1)[i].detach().cpu()
        #     nuclei_binary_tensor = torch.argmax(predictions_["nuclei_binary_map"], dim=-1)[i].detach().cpu() 
        #     hv_tensor = predictions_["hv_map"][i].detach().cpu()
            
        #     print(f"nuclei_type_tensor type: {type(nuclei_type_tensor)}, shape: {nuclei_type_tensor.shape}")
        #     print(f"nuclei_binary_tensor type: {type(nuclei_binary_tensor)}, shape: {nuclei_binary_tensor.shape}")
        #     print(f"hv_tensor type: {type(hv_tensor)}, shape: {hv_tensor.shape}")
            
        #     # 转换为 numpy
        #     nuclei_type = nuclei_type_tensor.numpy()[..., None]
        #     nuclei_binary = nuclei_binary_tensor.numpy()[..., None]
        #     hv = hv_tensor.numpy()
            
        #     print(f"nuclei_type type: {type(nuclei_type)}, shape: {nuclei_type.shape}")
        #     print(f"nuclei_binary type: {type(nuclei_binary)}, shape: {nuclei_binary.shape}")
        #     print(f"hv type: {type(hv)}, shape: {hv.shape}")
            
        #     # 然后尝试 concatenate
        #     pred_map = np.concatenate([nuclei_type, nuclei_binary, hv], axis=-1)
        for i in range(predictions_["nuclei_binary_map"].shape[0]):
            pred_map = np.concatenate(
                [
                    torch.argmax(predictions_["nuclei_type_map"], dim=-1)[i]
                    .detach()
                    .cpu().numpy()[..., None],
                    torch.argmax(predictions_["nuclei_binary_map"], dim=-1)[i]
                    .detach()
                    .cpu().numpy()[..., None],
                    predictions_["hv_map"][i].detach().cpu(),
                ],
                axis=-1,
            )
            instance_pred = cell_post_processor.post_process_cell_segmentation(pred_map)
            instance_preds.append(instance_pred[0])
            type_preds.append(instance_pred[1])

        return torch.Tensor(np.stack(instance_preds)), type_preds

    def generate_instance_nuclei_map(
        self, instance_maps: torch.Tensor, type_preds: List[dict]
    ) -> torch.Tensor:
        """Convert instance map (binary) to nuclei type instance map

        Args:
            instance_maps (torch.Tensor): Binary instance map, each instance has own integer. Shape: (B, H, W)
            type_preds (List[dict]): List (len=B) of dictionary with instance type information (compare post_process_hovernet function for more details)

        Returns:
            torch.Tensor: Nuclei type instance map. Shape: (B, self.num_nuclei_classes, H, W)
        """
        batch_size, h, w = instance_maps.shape
        instance_type_nuclei_maps = torch.zeros(
            (batch_size, h, w, self.num_nuclei_classes)
        )
        for i in range(batch_size):
            instance_type_nuclei_map = torch.zeros((h, w, self.num_nuclei_classes))
            instance_map = instance_maps[i]
            type_pred = type_preds[i]
            for nuclei, spec in type_pred.items():
                nuclei_type = spec["type"]
                instance_type_nuclei_map[:, :, nuclei_type][
                    instance_map == nuclei
                ] = nuclei

            instance_type_nuclei_maps[i, :, :, :] = instance_type_nuclei_map

        instance_type_nuclei_maps = instance_type_nuclei_maps.permute(0, 3, 1, 2)
        return torch.Tensor(instance_type_nuclei_maps)

    def freeze_encoder(self):
        """Freeze encoder to not train it"""
        for layer_name, p in self.encoder.named_parameters():
            if layer_name.split(".")[0] != "head":  # do not freeze head
                p.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder to train the whole model"""
        for p in self.encoder.parameters():
            p.requires_grad = True



class CrossAttentionModule(nn.Module):
    """
    双向交叉注意力模块：在 Transformer 输出的图像特征 (src) 之间实现信息互补
    """
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.attn_hv_to_seg = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_seg_to_hv = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_hv = nn.LayerNorm(dim)
        self.norm_seg = nn.LayerNorm(dim)

    def forward(self, feat_hv: torch.Tensor, feat_seg: torch.Tensor):
        """
        feat_hv, feat_seg: [B, C, H, W]
        """
        B, C, H, W = feat_hv.shape
        q_hv = feat_hv.flatten(2).transpose(1, 2) # [B, HW, C]
        q_seg = feat_seg.flatten(2).transpose(1, 2) # [B, HW, C]

        # Seg 从 HV 中提取几何边界信息
        out_seg, _ = self.attn_hv_to_seg(q_seg, q_hv, q_hv)
        q_seg = self.norm_seg(q_seg + out_seg)

        # HV 从 Seg 中提取语义区域信息
        out_hv, _ = self.attn_seg_to_hv(q_hv, q_seg, q_seg)
        q_hv = self.norm_hv(q_hv + out_hv)

        feat_hv = q_hv.transpose(1, 2).reshape(B, C, H, W)
        feat_seg = q_seg.transpose(1, 2).reshape(B, C, H, W)
        return feat_hv, feat_seg

class MLP(nn.Module):
    """SAM 风格的 MLP"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class CrossAttentionModule(nn.Module):
    """
    双向交叉注意力模块：在 Transformer 输出的图像特征 (src) 之间实现信息互补
    """
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.attn_hv_to_seg = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_seg_to_hv = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_hv = nn.LayerNorm(dim)
        self.norm_seg = nn.LayerNorm(dim)

    def forward(self, feat_hv: torch.Tensor, feat_seg: torch.Tensor):
        """
        feat_hv, feat_seg: [B, C, H, W]
        """
        B, C, H, W = feat_hv.shape
        q_hv = feat_hv.flatten(2).transpose(1, 2) # [B, HW, C]
        q_seg = feat_seg.flatten(2).transpose(1, 2) # [B, HW, C]

        # Seg 从 HV 中提取几何边界信息
        out_seg, _ = self.attn_hv_to_seg(q_seg, q_hv, q_hv)
        q_seg = self.norm_seg(q_seg + out_seg)

        # HV 从 Seg 中提取语义区域信息
        out_hv, _ = self.attn_seg_to_hv(q_hv, q_seg, q_seg)
        q_hv = self.norm_hv(q_hv + out_hv)

        feat_hv = q_hv.transpose(1, 2).reshape(B, C, H, W)
        feat_seg = q_seg.transpose(1, 2).reshape(B, C, H, W)
        return feat_hv, feat_seg

class MLP(nn.Module):
    """SAM 风格的 MLP"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class TokenGatingModule(nn.Module):
    """
    Predicts importance scores (Mask) for each token to filter out background noise.
    Optimized: Uses a high reduction ratio bottleneck or single layer to minimize parameters.
    """
    def __init__(self, embed_dim: int, reduction: int = 16):
        super().__init__()
        # Optimization: Reduce hidden dimension significantly (e.g., 1024 -> 64)
        hidden_dim = max(embed_dim // reduction, 16) # Ensure min dim of 16
        self.gating = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # Output [0, 1] probability
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C] - Patch tokens
        Returns: [B, N, 1] - Importance weights (Soft Mask)
        """
        return self.gating(x)

class CrossAttentionModule(nn.Module):
    """
    Bi-directional cross-attention for information exchange between task streams.
    """
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.attn_hv_to_seg = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_seg_to_hv = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_hv = nn.LayerNorm(dim)
        self.norm_seg = nn.LayerNorm(dim)

    def forward(self, feat_hv: torch.Tensor, feat_seg: torch.Tensor):
        B, C, H, W = feat_hv.shape
        q_hv = feat_hv.flatten(2).transpose(1, 2) 
        q_seg = feat_seg.flatten(2).transpose(1, 2) 

        out_seg, _ = self.attn_hv_to_seg(q_seg, q_hv, q_hv)
        q_seg = self.norm_seg(q_seg + out_seg)

        out_hv, _ = self.attn_seg_to_hv(q_hv, q_seg, q_seg)
        q_hv = self.norm_hv(q_hv + out_hv)

        feat_hv = q_hv.transpose(1, 2).reshape(B, C, H, W)
        feat_seg = q_seg.transpose(1, 2).reshape(B, C, H, W)
        return feat_hv, feat_seg

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class CellViTDINOv3(CellViT):
    """
    Optimized CellViTDINOv3 (Reduced Parameters) with Token Gating and Dual-Stream Interactive Decoding.
    """
    def __init__(
        self,
        model_path: Union[Path, str],
        num_nuclei_classes: int,
        num_tissue_classes: int,
        vit_structure: Literal["dinov3-s", "dinov3-b", "dinov3-l", "dinov3-h"],
        drop_rate: float = 0,
        regression_loss: bool = False,
        freeze_encoder: bool = True,
    ):
        self.model_path = model_path
        use_peft = True
        if vit_structure.lower() == "dinov3-s":
            self.init_dinov3_s(); self.model_name = "vit_small_plus_patch16_dinov3.lvd1689m"; use_peft = True
        elif vit_structure.lower() == "dinov3-b":
            self.init_dinov3_b(); self.model_name = "vit_base_patch16_dinov3.lvd1689m"; use_peft = True
        elif vit_structure.lower() == "dinov3-l":
            self.init_dinov3_l(); self.model_name = "vit_large_patch16_dinov3.lvd1689m"; use_peft =  True
        elif vit_structure.lower() == "dinov3-h":
            self.init_dinov3_h(); self.model_name = "vit_huge_plus_patch16_dinov3.lvd1689m"; use_peft = True
        else:
            raise NotImplementedError("Unknown DINOv3 structure")

        # Initialize parent class (CellViT)
        super().__init__(
            num_nuclei_classes=num_nuclei_classes, num_tissue_classes=num_tissue_classes,
            embed_dim=self.embed_dim, input_channels=3, depth=self.depth,
            num_heads=self.num_heads, extract_layers=self.extract_layers,
            mlp_ratio=0.25, qkv_bias=True, drop_rate=drop_rate, regression_loss=regression_loss,
        )

        # CLEANUP: Remove unused modules initialized by super().__init__ to save parameters
        # These are standard CellViT U-Net decoders which we are REPLACING with our SAM-based decoder
        unused_modules = [
            'decoder0', 'decoder1', 'decoder2', 'decoder3',
            'nuclei_binary_map_decoder', 'hv_map_decoder', 'nuclei_type_maps_decoder',
            'classifier_head' # Added: explicitly remove parent's classifier head if it exists
        ]
        for module_name in unused_modules:
            if hasattr(self, module_name):
                delattr(self, module_name)

        # Optimized Token Gating (reduction=16)
        self.token_gating = TokenGatingModule(self.embed_dim, reduction=16)

        # Overwrite encoder with DINOv3
        self.encoder = self.create_dinov3_encoder()
        if use_peft: 
            self.encoder.apply_peft(rank=64)
        elif freeze_encoder:
            self.logger_info("Freezing DINOv3 encoder weights (Linear Probing mode).")
            for param in self.encoder.parameters():
                param.requires_grad = False

        # REMOVED: self.classifier_head = nn.Linear(self.embed_dim, num_tissue_classes)

        self.projection = nn.Sequential(
            nn.Conv2d(self.embed_dim, 256, kernel_size=1, bias=False), LayerNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False), LayerNorm2d(256),
        )
        self.prompt_encoder = self.create_sam_prompt_encoder()

        # --- Decoder Optimization ---
        decoder_transformer_dim = 256
        decoder_mlp_dim = 256 
        decoder_depth = 2
        hyper_mlp_hidden = 256 

        
        
        # --- 分支 A: HV 相关 ---
        self.transformer_hv = TwoWayTransformer(
            depth=decoder_depth, 
            embedding_dim=decoder_transformer_dim, 
            mlp_dim=decoder_mlp_dim, 
            num_heads=8
        )
        self.mask_tokens_hv = nn.Embedding(2, 256) 
        self.hyper_mlps_hv = nn.ModuleList([
            MLP(256, hyper_mlp_hidden, 32, 3) for _ in range(2)
        ])
        
        # --- 分支 B: Segmentation 相关 ---
        self.transformer_seg = TwoWayTransformer(
            depth=decoder_depth, 
            embedding_dim=decoder_transformer_dim, 
            mlp_dim=decoder_mlp_dim, 
            num_heads=8
        )
        seg_out_channels = (2 + (2 if self.regression_loss else 0)) + self.num_nuclei_classes
        self.mask_tokens_seg = nn.Embedding(seg_out_channels + 1, 256)
        self.hyper_mlps_seg = nn.ModuleList([
            MLP(256, hyper_mlp_hidden, 32, 3) for _ in range(seg_out_channels + 1)
        ])

        self.iou_token = nn.Embedding(1, 256) 
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2), LayerNorm2d(64), nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), nn.GELU(),
        )
        self.cross_attention = CrossAttentionModule(dim=256, num_heads=8)
        self._init_dual_channel_splits()

    def _init_dual_channel_splits(self):
        offset = 2 if self.regression_loss else 0
        self.hv_splits = {'start': 0, 'end': 2}
        self.seg_splits = {
            'binary_start': 0, 'binary_end': 2 + offset,
            'type_start': 2 + offset, 'type_end': 2 + offset + self.num_nuclei_classes
        }

    def compute_gated_cosine_similarity(self, spatial_features: torch.Tensor, cls_token: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, Hp, Wp = spatial_features.shape
        tokens = spatial_features.flatten(2).transpose(1, 2) 
        
        token_weights = self.token_gating(tokens) # [B, N, 1]
        
        spatial_norm = F.normalize(tokens, p=2, dim=2) 
        cls_norm = F.normalize(cls_token, p=2, dim=1).unsqueeze(1) 
        sim = torch.bmm(spatial_norm, cls_norm.transpose(1, 2)) 
        
        gated_sim = token_weights * ((sim + 1.0) / 2.0)
        gated_sim_map = gated_sim.transpose(1, 2).view(B, 1, Hp, Wp)
        
        return gated_sim_map, token_weights

    def calculate_gating_loss(self, out_dict: dict, sparsity_weight: float = 0.01) -> torch.Tensor:
        if "token_weights" in out_dict:
            token_weights = out_dict["token_weights"]
            loss_sparsity = torch.mean(token_weights)
            return sparsity_weight * loss_sparsity
        return torch.tensor(0.0, device=self.projection[0].weight.device, requires_grad=True)

    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False) -> dict:
        out_dict = {}
        B, _, H, W = x.shape
        spatial_features, cls_token = self.encoder(x)
        # REMOVED: out_dict["tissue_types"] = self.classifier_head(cls_token)

        image_embeddings = self.projection(spatial_features)
        Hp, Wp = image_embeddings.shape[2:]

        gated_sim_map, token_weights = self.compute_gated_cosine_similarity(spatial_features, cls_token)
        out_dict["token_weights"] = token_weights 

        sim_prompt = F.interpolate(gated_sim_map, (H, W), mode="bilinear", align_corners=False)
        sparse_emb, dense_emb = self.prompt_encoder(points=None, boxes=None, masks=sim_prompt)
        
        if dense_emb.shape[2:] != (Hp, Wp):
            dense_emb = F.interpolate(dense_emb, size=(Hp, Wp), mode="bilinear", align_corners=False)
        image_pe = self.prompt_encoder.get_dense_pe()
        if image_pe.shape[2:] != (Hp, Wp):
            image_pe = F.interpolate(image_pe, size=(Hp, Wp), mode="bilinear", align_corners=False)

        tokens_hv = torch.cat([self.iou_token.weight, self.mask_tokens_hv.weight], dim=0).unsqueeze(0).expand(B, -1, -1)
        tokens_seg = torch.cat([self.iou_token.weight, self.mask_tokens_seg.weight], dim=0).unsqueeze(0).expand(B, -1, -1)
        
        src_hv = image_embeddings + dense_emb
        src_seg = image_embeddings + dense_emb

        hs_hv, updated_src_hv = self.transformer_hv(src_hv, image_pe, torch.cat([tokens_hv, sparse_emb], dim=1))
        hs_seg, updated_src_seg = self.transformer_seg(src_seg, image_pe, torch.cat([tokens_seg, sparse_emb], dim=1))

        feat_hv = updated_src_hv.transpose(1, 2).view(B, 256, Hp, Wp)
        feat_seg = updated_src_seg.transpose(1, 2).view(B, 256, Hp, Wp)

        feat_hv, feat_seg = self.cross_attention(feat_hv, feat_seg)

        out_dict["hv_map"] = self.predict_from_interactive_features(feat_hv, hs_hv, self.hyper_mlps_hv, (H, W))[:, self.hv_splits['start']:self.hv_splits['end']]
        
        seg_raw = self.predict_from_interactive_features(feat_seg, hs_seg, self.hyper_mlps_seg, (H, W))
        splits = self.seg_splits
        if self.regression_loss:
            out_dict["nuclei_binary_map"] = seg_raw[:, :2]
            out_dict["regression_map"] = seg_raw[:, 2:splits['binary_end']]
        else:
            out_dict["nuclei_binary_map"] = seg_raw[:, splits['binary_start']:splits['binary_end']]
        out_dict["nuclei_type_map"] = seg_raw[:, splits['type_start']:splits['type_end']]

        out_dict["cosine_sim_map"] = gated_sim_map
        return out_dict

    def predict_from_interactive_features(self, feat, hs, mlps, output_size):
        B, C, H, W = feat.shape
        upscaled = self.output_upscaling(feat) 
        mask_tokens_out = hs[:, 1 : (1 + len(mlps)), :]
        hyper_in_list = [mlps[i](mask_tokens_out[:, i, :]) for i in range(len(mlps))]
        hyper_in = torch.stack(hyper_in_list, dim=1) 
        b, c, h, w = upscaled.shape
        masks = (hyper_in @ upscaled.view(b, c, h * w)).view(b, -1, h, w)
        return F.interpolate(masks, size=output_size, mode="bilinear", align_corners=False)

    def create_sam_prompt_encoder(self):
        from segment_anything.modeling.prompt_encoder import PromptEncoder
        return PromptEncoder(embed_dim=256, image_embedding_size=(16, 16), input_image_size=(256, 256), mask_in_chans=16)

    def create_dinov3_encoder(self):
        return DinoV3BackboneWrapper(self.model_name, pretrained=True)

    def init_dinov3_s(self): self.embed_dim, self.depth, self.num_heads, self.extract_layers = 384, 12, 6, [2, 5, 8, 11]
    def init_dinov3_b(self): self.embed_dim, self.depth, self.num_heads, self.extract_layers = 768, 12, 12, [2, 5, 8, 11]
    def init_dinov3_l(self): self.embed_dim, self.depth, self.num_heads, self.extract_layers = 1024, 24, 16, [6, 12, 18, 24]
    def init_dinov3_h(self): self.embed_dim, self.depth, self.num_heads, self.extract_layers = 1280, 40, 24, [10, 20, 30, 40]

    def load_pretrained_encoder(self, model_path: str):
        pass

    def logger_info(self, msg):
        if hasattr(self, 'logger') and self.logger is not None: self.logger.info(msg)
        else: print(f"INFO: {msg}")



@dataclass
class DataclassHVStorage:
    """Storing PanNuke Prediction/GT objects for calculating loss, metrics etc. with HoverNet networks

    Args:
        nuclei_binary_map (torch.Tensor): Softmax output for binary nuclei branch. Shape: (batch_size, 2, H, W)
        hv_map (torch.Tensor): Logit output for HV-Map. Shape: (batch_size, 2, H, W)
        nuclei_type_map (torch.Tensor): Softmax output for nuclei type-prediction. Shape: (batch_size, num_tissue_classes, H, W)
        tissue_types (torch.Tensor): Logit tissue prediction output. Shape: (batch_size, num_tissue_classes)
        instance_map (torch.Tensor): Pixel-wise nuclear instance segmentation.
            Each instance has its own integer, starting from 1. Shape: (batch_size, H, W)
        instance_types_nuclei: Pixel-wise nuclear instance segmentation predictions, for each nuclei type.
            Each instance has its own integer, starting from 1.
            Shape: (batch_size, num_nuclei_classes, H, W)
        batch_size (int): Batch size of the experiment
        instance_types (list, optional): Instance type prediction list.
            Each list entry stands for one image. Each list entry is a dictionary with the following structure:
            Main Key is the nuclei instance number (int), with a dict as value.
            For each instance, the dictionary contains the keys: bbox (bounding box), centroid (centroid coordinates),
            contour, type_prob (probability), type (nuclei type)
            Defaults to None.
        regression_map (torch.Tensor, optional): Regression map for binary prediction map.
            Shape: (batch_size, 2, H, W). Defaults to None.
        regression_loss (bool, optional): Indicating if regression map is present. Defaults to False.
        h (int, optional): Height of used input images. Defaults to 256.
        w (int, optional): Width of used input images. Defaults to 256.
        num_tissue_classes (int, optional): Number of tissue classes in the data. Defaults to 19.
        num_nuclei_classes (int, optional): Number of nuclei types in the data (including background). Defaults to 6.
    """

    nuclei_binary_map: torch.Tensor
    hv_map: torch.Tensor
    tissue_types: torch.Tensor
    nuclei_type_map: torch.Tensor
    instance_map: torch.Tensor
    instance_types_nuclei: torch.Tensor
    batch_size: int
    instance_types: list = None
    regression_map: torch.Tensor = None
    regression_loss: bool = False
    h: int = 256
    w: int = 256
    num_tissue_classes: int = 19
    num_nuclei_classes: int = 6

    # def __post_init__(self):
    #     # check shape of every element
    #     assert list(self.nuclei_binary_map.shape) == [
    #         self.batch_size,
    #         2,
    #         self.h,
    #         self.w,
    #     ], "Nuclei Binary Map must be a softmax tensor with shape (B, 2, H, W)"
    #     assert list(self.hv_map.shape) == [
    #         self.batch_size,
    #         2,
    #         self.h,
    #         self.w,
    #     ], "HV Map must be a tensor with shape (B, 2, H, W)"
    #     assert list(self.nuclei_type_map.shape) == [
    #         self.batch_size,
    #         self.num_nuclei_classes,
    #         self.h,
    #         self.w,
    #     ], "Nuclei Type Map must be a tensor with shape (B, num_nuclei_classes, H, W)"
    #     assert list(self.instance_map.shape) == [
    #         self.batch_size,
    #         self.h,
    #         self.w,
    #     ], "Instance Map must be a tensor with shape (B, H, W)"
    #     assert list(self.instance_types_nuclei.shape) == [
    #         self.batch_size,
    #         self.num_nuclei_classes,
    #         self.h,
    #         self.w,
    #     ], "Instance Types Nuclei must be a tensor with shape (B, num_nuclei_classes, H, W)"
    #     if self.regression_map is not None:
    #         self.regression_loss = True
    #     else:
    #         self.regression_loss = False

    def get_dict(self) -> dict:
        """Return dictionary of entries"""
        property_dict = self.__dict__
        if not self.regression_loss and "regression_map" in property_dict.keys():
            property_dict.pop("regression_map")
        return property_dict