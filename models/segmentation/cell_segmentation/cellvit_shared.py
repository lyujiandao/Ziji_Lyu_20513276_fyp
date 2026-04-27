# -*- coding: utf-8 -*-
# CellViT networks and adaptions, with shared encoders
#
# UNETR paper and code: https://github.com/tamasino52/UNETR
# SAM paper and code: https://segment-anything.com/
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import List, Literal, Union
from models.mlp.archs import DWConv, shiftedBlock, shiftmlp

import torch
import torch.nn as nn

from .cellvit import CellViT
from .utils import Conv2DBlock, Deconv2DBlock, ViTCellViT, ViTCellViTDeit
from segment_anything import sam_model_registry


class CellViTShared(CellViT, nn.Module):
    """CellViT Modell for cell segmentation. U-Net like network with vision transformer as backbone encoder

    All heads are shared, just final layers are not shared

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
        nn.Module.__init__(self)
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
        self.regression_loss = regression_loss

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

        offset_branches = 0
        if self.regression_loss:
            offset_branches = 2
        self.branches_output = {
            "nuclei_binary_map": 2 + offset_branches,
            "hv_map": 2,
            "nuclei_type_maps": self.num_nuclei_classes,
        }

        self.decoder = self.create_upsampling_branch()
        self.nuclei_binary_map_decoder = nn.Conv2d(
            in_channels=64,
            out_channels=2 + offset_branches,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.hv_map_decoder = nn.Conv2d(
            in_channels=64,
            out_channels=2,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.nuclei_type_maps_decoder = nn.Conv2d(
            in_channels=64,
            out_channels=self.num_nuclei_classes,
            kernel_size=1,
            stride=1,
            padding=0,
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
                * (optional) tokens
                * (optional) regression_map
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

        upsampled = self._forward_upsample(z0, z1, z2, z3, z4, self.decoder)
        if self.regression_loss:
            nb_map = self.nuclei_binary_map_decoder(upsampled)
            out_dict["nuclei_binary_map"] = nb_map[:, :2, :, :]
            out_dict["regression_map"] = nb_map[:, 2:, :, :]
        else:
            out_dict["nuclei_binary_map"] = self.nuclei_binary_map_decoder(upsampled)
        out_dict["hv_map"] = self.hv_map_decoder(upsampled)
        out_dict["nuclei_type_map"] = self.nuclei_type_maps_decoder(upsampled)

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
        b3 = branch_decoder.decoder3_skip(z3)
        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        b2 = branch_decoder.decoder2_skip(z2)
        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))
        b1 = branch_decoder.decoder1_skip(z1)
        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        b0 = branch_decoder.decoder0_skip(z0)
        b_final = branch_decoder.decoder0_header(torch.cat([b0, b1], dim=1))

        return b_final

    def create_upsampling_branch(self) -> nn.Module:
        """Create Upsampling branch

        Returns:
            nn.Module: Upsampling path
        """
        # Skip connections
        decoder0_skip = nn.Sequential(
            Conv2DBlock(3, 32, 3, self.drop_rate),
            Conv2DBlock(32, 64, 3, self.drop_rate),
        )  # skip connection after positional encoding, shape should be H, W, 64
        decoder1_skip = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_12, 128, dropout=self.drop_rate),
        )  # skip connection 1
        decoder2_skip = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, 256, dropout=self.drop_rate),
        )  # skip connection 2
        decoder3_skip = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate)
        )  # skip connection 3

        # Upsampling
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
        )

        decoder = nn.Sequential(
            OrderedDict(
                [
                    ("decoder0_skip", decoder0_skip),
                    ("decoder1_skip", decoder1_skip),
                    ("decoder2_skip", decoder2_skip),
                    ("decoder3_skip", decoder3_skip),
                    ("bottleneck_upsampler", bottleneck_upsampler),
                    ("decoder3_upsampler", decoder3_upsampler),
                    ("decoder2_upsampler", decoder2_upsampler),
                    ("decoder1_upsampler", decoder1_upsampler),
                    ("decoder0_header", decoder0_header),
                ]
            )
        )

        return decoder


class CellViT256Shared(CellViTShared):
    """CellViT with ViT-256 backbone settings (https://github.com/mahmoodlab/HIPT/blob/master/HIPT_4K/Checkpoints/vit256_small_dino.pth)

    All heads are shared, just final layers are not shared

    Args:
        model256_path (Union[Path, str]): Path to ViT 256 backbone model
        num_nuclei_classes (int): Number of nuclei classes (including background)
        num_tissue_classes (int): Number of tissue classes
        drop_rate (float, optional): Dropout in MLP. Defaults to 0.
        attn_drop_rate (float, optional): Dropout for attention layer in backbone ViT. Defaults to 0.
        drop_path_rate (float, optional): Dropout for skip connection . Defaults to 0.
        regression_loss (bool, optional): Use regressive loss for predicting vector components.
            Adds two additional channels to the binary decoder, but returns it as own entry in dict. Defaults to False.
    """

    def __init__(
        self,
        model256_path: Union[Path, str],
        num_nuclei_classes: int,
        num_tissue_classes: int,
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
        regression_loss: bool = False,
    ):
        self.patch_size = 16
        self.embed_dim = 384
        self.depth = 12
        self.num_heads = 6
        self.mlp_ratio = 4
        self.qkv_bias = True
        self.extract_layers = [3, 6, 9, 12]
        self.input_channels = 3  # RGB
        self.num_tissue_classes = num_tissue_classes
        self.num_nuclei_classes = num_nuclei_classes

        super().__init__(
            num_nuclei_classes,
            num_tissue_classes,
            self.embed_dim,
            self.input_channels,
            self.depth,
            self.num_heads,
            self.extract_layers,
            self.mlp_ratio,
            self.qkv_bias,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            regression_loss,
        )

        self.model256_path = model256_path

    def load_pretrained_encoder(self, model256_path):
        state_dict = torch.load(str(model256_path), map_location="cpu")["teacher"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = self.encoder.load_state_dict(state_dict, strict=False)
        print(f"Loading checkpoint: {msg}")


class CellViTSAMShared(CellViTShared):
    """CellViT with SAM backbone settings

    All heads are shared, just final layers are not shared

    Args:
        model_path (Union[Path, str]): Path to pretrained SAM model
        num_nuclei_classes (int): Number of nuclei classes (including background)
        num_tissue_classes (int): Number of tissue classes
        vit_structure (Literal["SAM-B", "SAM-L", "SAM-H"]): SAM model type
        drop_rate (float, optional): Dropout in MLP. Defaults to 0.
        regression_loss (bool, optional): Use regressive loss for predicting vector components.
            Adds two additional channels to the binary decoder, but returns it as own entry in dict. Defaults to False.

    Raises:
        NotImplementedError: Unknown SAM configuration
    """

    def __init__(
        self,
        model_path: Union[Path, str],
        num_nuclei_classes: int,
        num_tissue_classes: int,
        vit_structure: Literal["SAM-B", "SAM-L", "SAM-H"],
        drop_rate: float = 0,
        regression_loss: bool = False,
    ):
        if vit_structure.upper() == "SAM-B":
            self.init_vit_b()
        elif vit_structure.upper() == "SAM-L":
            self.init_vit_l()
        elif vit_structure.upper() == "SAM-H":
            self.init_vit_h()
        else:
            raise NotImplementedError("Unknown ViT-SAM backbone structure")

        self.input_channels = 3  # RGB
        self.mlp_ratio = 4
        self.qkv_bias = True
        self.model_path = model_path

        super().__init__(
            num_nuclei_classes=num_nuclei_classes,
            num_tissue_classes=num_tissue_classes,
            embed_dim=self.embed_dim,
            input_channels=self.input_channels,
            depth=self.depth,
            num_heads=self.num_heads,
            extract_layers=self.extract_layers,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            drop_rate=drop_rate,
            regression_loss=regression_loss,
        )

        self.prompt_embed_dim = 256

        self.encoder = ViTCellViTDeit(
            extract_layers=self.extract_layers,
            depth=self.depth,
            embed_dim=self.embed_dim,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=self.num_heads,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=self.encoder_global_attn_indexes,
            window_size=14,
            out_chans=self.prompt_embed_dim,
        )

        self.classifier_head = (
            nn.Linear(self.prompt_embed_dim, num_tissue_classes)
            if num_tissue_classes > 0
            else nn.Identity()
        )

    def load_pretrained_encoder(self, model_path):
        """Load pretrained SAM encoder from provided path

        Args:
            model_path (str): Path to SAM model
        """
        state_dict = torch.load(str(model_path), map_location="cpu")
        image_encoder = self.encoder
        msg = image_encoder.load_state_dict(state_dict, strict=False)
        print(f"Loading checkpoint: {msg}")
        self.encoder = image_encoder

    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False):
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
        ), "Img must have a shape of that is divisble by patch_soze (token_size)"
        assert (
            x.shape[-1] % self.patch_size == 0
        ), "Img must have a shape of that is divisble by patch_soze (token_size)"

        out_dict = {}

        classifier_logits, _, z = self.encoder(x)
        out_dict["tissue_types"] = self.classifier_head(classifier_logits)

        z0, z1, z2, z3, z4 = x, *z

        # performing reshape for the convolutional layers and upsampling (restore spatial dimension)
        z4 = z4.permute(0, 3, 1, 2)
        z3 = z3.permute(0, 3, 1, 2)
        z2 = z2.permute(0, 3, 1, 2)
        z1 = z1.permute(0, 3, 1, 2)

        upsampled = self._forward_upsample(z0, z1, z2, z3, z4, self.decoder)
        if self.regression_loss:
            nb_map = self.nuclei_binary_map_decoder(upsampled)
            out_dict["nuclei_binary_map"] = nb_map[:, :2, :, :]
            out_dict["regression_map"] = nb_map[:, 2:, :, :]
        else:
            out_dict["nuclei_binary_map"] = self.nuclei_binary_map_decoder(upsampled)

        out_dict["hv_map"] = self.hv_map_decoder(upsampled)
        out_dict["nuclei_type_map"] = self.nuclei_type_maps_decoder(upsampled)

        if retrieve_tokens:
            out_dict["tokens"] = z4

        return out_dict

    def init_vit_b(self):
        self.embed_dim = 768
        self.depth = 12
        self.num_heads = 12
        self.encoder_global_attn_indexes = [2, 5, 8, 11]
        self.extract_layers = [3, 6, 9, 12]

    def init_vit_l(self):
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 16
        self.encoder_global_attn_indexes = [5, 11, 17, 23]
        self.extract_layers = [6, 12, 18, 24]

    def init_vit_h(self):
        self.embed_dim = 1280
        self.depth = 32
        self.num_heads = 16
        self.encoder_global_attn_indexes = [7, 15, 23, 31]
        self.extract_layers = [8, 16, 24, 32]



class CellViTDINOv3Shared(CellViTShared):
    """CellViT with DINOv3 backbone settings (shared decoders)

    All heads are shared, just final layers are not shared

    Args:
        model_path (Union[Path, str]): Path to DINOv3 backbone model
        num_nuclei_classes (int): Number of nuclei classes (including background)
        num_tissue_classes (int): Number of tissue classes
        vit_structure (Literal["dinov3-s", "dinov3-b", "dinov3-l", "dinov3-g"]): DINOv3 model type
        drop_rate (float, optional): Dropout in MLP. Defaults to 0.
        regression_loss (bool, optional): Use regressive loss for predicting vector components.
            Adds two additional channels to the binary decoder, but returns it as own entry in dict. Defaults to False.
    """

    def __init__(
        self,
        model_path: Union[Path, str],
        num_nuclei_classes: int,
        num_tissue_classes: int,
        vit_structure: Literal["dinov3-s", "dinov3-b", "dinov3-l", "dinov3-g"],
        drop_rate: float = 0,
        regression_loss: bool = False,
    ):
        # 根据不同的DINOv3结构初始化参数
        if vit_structure.lower() == "dinov3-s":
            self.init_dinov3_s()
        elif vit_structure.lower() == "dinov3-b":
            self.init_dinov3_b()
        elif vit_structure.lower() == "dinov3-l":
            self.init_dinov3_l()
        elif vit_structure.lower() == "dinov3-g":
            self.init_dinov3_g()
        else:
            raise NotImplementedError("Unknown DINOv3 backbone structure")

        self.input_channels = 3  # RGB
        self.mlp_ratio = 4
        self.qkv_bias = True
        self.model_path = model_path

        super().__init__(
            num_nuclei_classes=num_nuclei_classes,
            num_tissue_classes=num_tissue_classes,
            embed_dim=self.embed_dim,
            input_channels=self.input_channels,
            depth=self.depth,
            num_heads=self.num_heads,
            extract_layers=self.extract_layers,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            drop_rate=drop_rate,
            regression_loss=regression_loss,
        )

        self.classifier_head = nn.Linear(self.embed_dim, num_tissue_classes)

        # 替换encoder为DINOv3兼容的版本
        self.encoder = self.create_dinov3_encoder()
        self.wrap_dinov3_blocks_with_adapters(freeze_backbone= True)

        # self.test_dimension_compatibility()

    def wrap_dinov3_blocks_with_adapters(self, freeze_backbone=True):
        # 获取DINOv3 encoder的blocks
        # print(f"myencoder{dir(self.encoder.named_parameters)}")
        # print(f"encoder{dir(self.encoder.timm_model)}")
        if hasattr(self.encoder.timm_model, 'blocks'):
            # transformers库的标准结构
            blocks = self.encoder.timm_model.blocks

        
        adapted_blocks = []
        for block in blocks:
            adapted_block = DINOv3Adapter(block)  # 创建Adapter包装器
            adapted_blocks.append(adapted_block)
        
        # print(f"lengh of apapted_blocks{len(adapted_blocks)}")

 


        if hasattr(self.encoder.timm_model, 'blocks'):
            self.encoder.timm_model.blocks = nn.ModuleList(adapted_blocks)
        

        if freeze_backbone:
            for name, param in self.encoder.timm_model.named_parameters():
                if 'prompt_learn' not in name and 'adapter_mlp' not in name:  # 只训练Adapter参数
                    param.requires_grad = False

        # print(f"Wrapped {len(adapted_blocks)} DINOv3 blocks with adapters")

    def create_upsampling_branch(self) -> nn.Module:
            """为DINOv3创建适配的上采样分支"""
            # 为DINOv3设置正确的维度
            if self.embed_dim == 384:  # DINOv3-small
                self.skip_dim_11 = 256
                self.skip_dim_12 = 128
                self.bottleneck_dim = 384  # 关键：匹配384维特征
            elif self.embed_dim == 768:  # DINOv3-base
                self.skip_dim_11 = 512
                self.skip_dim_12 = 256  
                self.bottleneck_dim = 512
            elif self.embed_dim == 1024:  # DINOv3-large
                self.skip_dim_11 = 512
                self.skip_dim_12 = 256
                self.bottleneck_dim = 512
            elif self.embed_dim == 1536:  # DINOv3-giant
                self.skip_dim_11 = 768
                self.skip_dim_12 = 384
                self.bottleneck_dim = 768
                

            
            # 调用父类方法创建实际的解码器结构
            return super().create_upsampling_branch()

    def create_dinov3_encoder(self):
        """创建DINOv3编码器"""
        from transformers import AutoModel
        

        
        # print(f"self.model_path: {self.model_path}")
        
        backbone_model = AutoModel.from_pretrained(
            "backbone",
            torch_dtype=torch.float32,
            local_files_only = True
        )
        return backbone_model

    def load_pretrained_encoder(self, model_path: str):
        """加载预训练的DINOv3编码器"""
        pass


    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False) -> dict:
        """DINOv3专用的前向传播"""
        assert (
            x.shape[-2] % self.patch_size == 0
        ), f"Image height must be divisible by patch_size {self.patch_size}"
        assert (
            x.shape[-1] % self.patch_size == 0
        ), f"Image width must be divisible by patch_size {self.patch_size}"

        out_dict = {}

        # DINOv3前向传播 - 获取所有隐藏层
        outputs = self.encoder(x, output_hidden_states=True)
        
        # 提取隐藏状态 - 已经是空间格式 [B, C, H, W]
        hidden_states = outputs.hidden_states
        
        # print(f"Debug: hidden_states长度 = {len(hidden_states)}")  # 应该是13（包括patch嵌入）或12
        
        # 根据extract_layers提取特定层特征
        # hidden_states[0]可能是patch嵌入，[1]是第一个Transformer块输出
        # 对于12层的DINOv3-small，索引应该是0-11
        z1 = hidden_states[self.extract_layers[0]]  # 第3层
        z2 = hidden_states[self.extract_layers[1]]  # 第6层  
        z3 = hidden_states[self.extract_layers[2]]  # 第9层
        z4 = hidden_states[self.extract_layers[3]]  # 第12层（最后一层）
        
        print(f"Debug: 提取的特征形状:")
        print(f"  z0: {z0.shape}")
        print(f"  z1: {z1.shape}")
        print(f"  z2: {z2.shape}") 
        print(f"  z3: {z3.shape}")
        print(f"  z4: {z4.shape}")
        
        # 获取分类特征 - 使用pooler_output
        classifier_logits = outputs.pooler_output  # 形状: [B, 384]
        out_dict["tissue_types"] = self.classifier_head(classifier_logits)

        z0 = x  # 原始输入 [B, 3, 256, 256]

        # DINOv3的输出已经是空间格式，不需要形状转换！
        # z1-z4形状: [B, 384, 16, 16] - 可以直接用于解码器
        print(f"z4:{z4.shape}")
        # 使用共享解码器进行上采样
        upsampled = self._forward_upsample(z0, z1, z2, z3, z4, self.decoder)
        
        print(f"Debug: 上采样后形状: {upsampled.shape}")
        
        # 各个任务头
        if self.regression_loss:
            nb_map = self.nuclei_binary_map_decoder(upsampled)
            out_dict["nuclei_binary_map"] = nb_map[:, :2, :, :]
            out_dict["regression_map"] = nb_map[:, 2:, :, :]
        else:
            out_dict["nuclei_binary_map"] = self.nuclei_binary_map_decoder(upsampled)
        
        out_dict["hv_map"] = self.hv_map_decoder(upsampled)
        out_dict["nuclei_type_map"] = self.nuclei_type_maps_decoder(upsampled)

        if retrieve_tokens:
            # 使用最后一层的空间特征作为tokens
            out_dict["tokens"] = z4


        a = out_dict["nuclei_binary_map"]
        print(f"out_dict:{a.shape}")
        return out_dict



    # def test_dimension_compatibility(self):
    #     """测试模型维度兼容性"""
    #     try:
    #         # 创建测试输入
    #         x = torch.randn(1, 3, 256, 256)
            
    #         print("1. 测试编码器...")
    #         with torch.no_grad():
    #             outputs = self.encoder(x, output_hidden_states=True)
    #             hidden_states = outputs.hidden_states
                
    #             # 调试：查看hidden_states的实际结构
    #             print(f"   hidden_states长度: {len(hidden_states)}")
    #             for i, state in enumerate(hidden_states):
    #                 print(f"   层 {i}: {state.shape}")
                
    #             # 检查extract_layers是否有效
    #             print(f"   extract_layers配置: {self.extract_layers}")
    #             for layer_idx in self.extract_layers:
    #                 if layer_idx >= len(hidden_states):
    #                     print(f"   ❌ 错误: extract_layers[{layer_idx}] 超出范围 (最大索引: {len(hidden_states)-1})")
    #                     # 自动调整extract_layers
    #                     self._adjust_extract_layers(len(hidden_states))
    #                     break
                
    #             # 重新提取特征
    #             z1 = hidden_states[self.extract_layers[0]]
    #             z2 = hidden_states[self.extract_layers[1]]  
    #             z3 = hidden_states[self.extract_layers[2]]
    #             z4 = hidden_states[self.extract_layers[3]]
                
    #             print(f"   编码器特征形状:")
    #             print(f"     z1 (层{self.extract_layers[0]}): {z1.shape}")
    #             print(f"     z2 (层{self.extract_layers[1]}): {z2.shape}")
    #             print(f"     z3 (层{self.extract_layers[2]}): {z3.shape}")
    #             print(f"     z4 (层{self.extract_layers[3]}): {z4.shape}")
            
    #         print("2. 测试解码器各层...")
    #         # 测试解码器各层
    #         self._test_decoder_layers(z1, z2, z3, z4, x)
            
    #         print("✅ 所有维度测试通过")
            
    #     except Exception as e:
    #         print(f"❌ 维度测试失败: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         raise

    # def _adjust_extract_layers(self, total_layers):
    #     """根据实际层数调整extract_layers"""
    #     print(f"   自动调整extract_layers...")
    #     print(f"   总层数: {total_layers}, 原extract_layers: {self.extract_layers}")
        
    #     # 均匀选择4个层
    #     step = total_layers // 4
    #     new_extract_layers = [step, step*2, step*3, total_layers-1]  # 最后一个用最后一层
        
    #     # 确保索引不越界
    #     new_extract_layers = [min(idx, total_layers-1) for idx in new_extract_layers]
        
    #     self.extract_layers = new_extract_layers
    #     print(f"   新extract_layers: {self.extract_layers}")
    
    # def _test_decoder_layers(self, z1, z2, z3, z4, z0):
    #     """测试解码器各层维度"""
    #     # 瓶颈上采样
    #     print("   测试bottleneck_upsampler...")
    #     b4 = self.decoder.bottleneck_upsampler(z4)
    #     print(f"     {z4.shape} -> {b4.shape}")
        
    #     # decoder3_skip + upsampler
    #     print("   测试decoder3...")
    #     b3_skip = self.decoder.decoder3_skip(z3)
    #     b3_input = torch.cat([b3_skip, b4], dim=1)
    #     b3 = self.decoder.decoder3_upsampler(b3_input)
    #     print(f"     {z3.shape} -> {b3_skip.shape} + {b4.shape} = {b3_input.shape} -> {b3.shape}")
        
    #     # decoder2_skip + upsampler  
    #     print("   测试decoder2...")
    #     b2_skip = self.decoder.decoder2_skip(z2)
    #     b2_input = torch.cat([b2_skip, b3], dim=1)
    #     b2 = self.decoder.decoder2_upsampler(b2_input)
    #     print(f"     {z2.shape} -> {b2_skip.shape} + {b3.shape} = {b2_input.shape} -> {b2.shape}")
        
    #     # decoder1_skip + upsampler
    #     print("   测试decoder1...")
    #     b1_skip = self.decoder.decoder1_skip(z1)
    #     b1_input = torch.cat([b1_skip, b2], dim=1)
    #     b1 = self.decoder.decoder1_upsampler(b1_input)
    #     print(f"     {z1.shape} -> {b1_skip.shape} + {b2.shape} = {b1_input.shape} -> {b1.shape}")
        
    #     # decoder0_skip + header
    #     print("   测试decoder0...")
    #     b0_skip = self.decoder.decoder0_skip(z0)
    #     b0_input = torch.cat([b0_skip, b1], dim=1)
    #     b_final = self.decoder.decoder0_header(b0_input)
    #     print(f"     {z0.shape} -> {b0_skip.shape} + {b1.shape} = {b0_input.shape} -> {b_final.shape}")

    def init_dinov3_s(self):
        self.embed_dim = 384
        self.depth = 12
        self.num_heads = 6
        self.extract_layers = [2, 5, 8, 11]
        self.patch_size = 16

    def init_dinov3_b(self):
        self.embed_dim = 768
        self.depth = 12
        self.num_heads = 12
        self.extract_layers = [2, 5, 8, 11]
        self.patch_size = 16

    def init_dinov3_l(self):
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 16
        self.extract_layers = [6, 12, 18, 24]
        self.patch_size = 16

    def init_dinov3_g(self):
        self.embed_dim = 1536
        self.depth = 40
        self.num_heads = 24
        self.extract_layers = [10, 20, 30, 40]
        self.patch_size = 16


class DINOv3Adapter(nn.Module):
    """DINOv3的Adapter包装器使用archs.py中的shiftedBlock"""
    
    def __init__(self, block):
        super().__init__()
        self.block = block
        
        # 获取block的输入维度
        if hasattr(block, 'attention') and hasattr(block.attention, 'qkv'):
            dim = block.attention.qkv.in_features
        elif hasattr(block, 'attn') and hasattr(block.attn, 'qkv'):
            dim = block.attn.qkv.in_features
        else:
            dim = 768
        
        self.dim = dim

        # 1 for simple, 2 for block, 3 for mlp
        self.mymode = 1


        if self.mymode ==1:
            reduction = 16
            
            # Adapter网络结构
            self.adapter_block = nn.Sequential(
                nn.Linear(dim, dim // reduction),
                nn.GELU(),
                nn.Linear(dim // reduction, dim),
                nn.GELU()
            )
        elif self.mymode ==2:
            self.adapter_block = shiftedBlock(
                dim=dim,
                num_heads=4, 
                mlp_ratio=4,
                drop=0.1,
                attn_drop=0.1,
                drop_path=0.1,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm
            )
        else:
            self.adapter_block = shiftmlp(
                in_features=dim,
                hidden_features=dim * 4,
                out_features=dim,
                act_layer=nn.GELU,
                drop=0.1,
                shift_size=5
            )
    



    def forward(self, x, rope=None):
        """
        🔥 关键修改：参考 adapter_encoder.py
        在 norm2 之后，Adapter 与 MLP 并行
        """
        
        # ===== 第一部分：Attention 分支 =====
        shortcut_attn = x
        
        # Norm1
        if hasattr(self.block, 'norm1') or hasattr(self.block, 'ln1'):
            # print("norm1")
            norm1 = self.block.norm1 if hasattr(self.block, 'norm1') else self.block.ln1
            x = norm1(x)
        
        # Attention
        if hasattr(self.block, 'attn'):
            # print("attn")
            x = self.block.attn(x)
        elif hasattr(self.block, 'attention'):
            print("attention")
            x = self.block.attention(x)
        
        # 残差连接
        x = shortcut_attn + x
        
        
        # ===== 第二部分：MLP + Adapter 并行分支 =====
        shortcut_mlp = x
        
        # 🔥 Norm2（就是在这里！）
        if hasattr(self.block, 'norm2') or hasattr(self.block, 'ln2'):
            # print("norm2")
            norm2 = self.block.norm2 if hasattr(self.block, 'norm2') else self.block.ln2
            xn = norm2(x)
        else:
            xn = x
        
        # 🔥 原始 MLP 分支（冻结）
        if hasattr(self.block, 'mlp'):
            # print("mlp")
            x_mlp = self.block.mlp(xn)
        else:
            x_mlp = torch.zeros_like(xn)
        
        # 🔥 Adapter 分支（可训练）
        # 处理 DINOv3 的 token 结构
        if xn.dim() == 3:  # [B, N, C]
            B, N, C = xn.shape
            
            # 分离 image token 和 special token
            if N > 256:
                image_token = xn[:, :256, :]
                special_token = xn[:, 256:, :]
                
                if self.mymode == 1:
                    adapter_out_image = self.adapter_block(image_token)
                else:
                    H = W = 16
                    adapter_out_image = self.adapter_block(image_token, H, W)

                
                # 重新拼接
                adapter_out = torch.cat([adapter_out_image, special_token], dim=1)
            else:
                print("Somethingerror")
                H = W = int(xn.sqrt(N))
                adapter_out = self.adapter_block(xn, H, W)
        else:
            print("Noooooooooooooooooooooo")
            adapter_out = xn
        
        # 🔥 并行相加（参考 adapter_encoder.py）

        x = xn + x_mlp + 0.5 * adapter_out

        return x


class DINOv3Adapter2(nn.Module):
    def __init__(self, blk):
        super().__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        reduction = 16
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.GELU(),
            nn.Linear(dim // reduction, dim),
            nn.GELU()
        )
    
    def forward(self, x, rope=None):
        prompt = self.prompt_learn(x)
        prompted = x + prompt
        return self.block(prompted, rope=rope)

# class CellViTDINOv3Shared(CellViTShared):
#     """CellViT with DINOv3 backbone and official SAM decoder

#     Uses DINOv3 as encoder and SAM's official mask decoder

#     Args:
#         model_path (Union[Path, str]): Path to DINOv3 backbone model
#         num_nuclei_classes (int): Number of nuclei classes (including background)
#         num_tissue_classes (int): Number of tissue classes
#         vit_structure (Literal["dinov3-s", "dinov3-b", "dinov3-l", "dinov3-g"]): DINOv3 model type
#         sam_checkpoint (Union[Path, str]): Path to SAM model checkpoint (用于加载解码器权重)
#         drop_rate (float, optional): Dropout in MLP. Defaults to 0.
#         regression_loss (bool, optional): Use regressive loss for predicting vector components.
#             Adds two additional channels to the binary decoder, but returns it as own entry in dict. Defaults to False.
#     """

#     def __init__(
#         self,
#         model_path: Union[Path, str],
#         num_nuclei_classes: int,
#         num_tissue_classes: int,
#         vit_structure: Literal["dinov3-s", "dinov3-b", "dinov3-l", "dinov3-g"],
#         drop_rate: float = 0,
#         regression_loss: bool = False,
#     ):
#         # 根据不同的DINOv3结构初始化参数
#         if vit_structure.lower() == "dinov3-s":
#             self.init_dinov3_s()
#         elif vit_structure.lower() == "dinov3-b":
#             self.init_dinov3_b()
#         elif vit_structure.lower() == "dinov3-l":
#             self.init_dinov3_l()
#         elif vit_structure.lower() == "dinov3-g":
#             self.init_dinov3_g()
#         else:
#             raise NotImplementedError("Unknown DINOv3 backbone structure")
        

#         self.input_channels = 3  # RGB
#         self.mlp_ratio = 0.25
#         self.qkv_bias = True
#         self.model_path = model_path
        

#         super().__init__(
#             num_nuclei_classes=num_nuclei_classes,
#             num_tissue_classes=num_tissue_classes,
#             embed_dim=self.embed_dim,
#             input_channels=self.input_channels,
#             depth=self.depth,
#             num_heads=self.num_heads,
#             extract_layers=self.extract_layers,
#             mlp_ratio=self.mlp_ratio,
#             qkv_bias=self.qkv_bias,
#             drop_rate=drop_rate,
#             regression_loss=regression_loss,
#             )
        

#         # self.projection = nn.Conv2d(384,256,kernel_size = 1).to("cuda:1")  # DINOv3的最后输出通道是384，而SAM解码器需要256
#         self.projection = nn.Sequential(
#             # 第一步: 384→320
#             nn.Conv2d(384, 320, kernel_size=1),
#             nn.BatchNorm2d(320),
#             nn.ReLU(inplace=True),
        
#             # 第二步: 320→256
#             nn.Conv2d(320, 256, kernel_size=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#         ).to("cuda:1")



#         self.layer_weights = nn.Parameter(torch.ones(4) / 4)
#         self.num_nuclei_classes = num_nuclei_classes
#         self.num_tissue_classes = num_tissue_classes
#         self.regression_loss = regression_loss
#         self.drop_rate = drop_rate

#         # SAM解码器需要image_pe（位置编码）
#         from segment_anything.modeling.prompt_encoder import PositionEmbeddingRandom
#         self.pe_layer = PositionEmbeddingRandom(128)  # 128 是位置编码的维度
#         self._image_pe_cache = None # 假设特征图大小是 64x64


#         # DINOv3编码器
#         self.encoder = self.create_dinov3_encoder()
#         self.wrap_dinov3_blocks_with_adapters(freeze_backbone= True)
#         self.classifier_head = nn.Linear(self.embed_dim, num_tissue_classes)

#         #MLP
#         # self.mlp_connection = self.create_mlp_connection()

#         # SAM解码器组件
#         self.prompt_encoder = self.create_sam_prompt_encoder("./models/sam/sam_vit_h_4b8939.pth").to("cuda:1")
#         self.sam_decoder = self.create_sam_decoder("./models/sam/sam_vit_h_4b8939.pth").to("cuda:1")
        
#         # 输出投影层，将SAM解码器输出映射到我们的多任务输出
#         # self.output_projection = self.create_output_projection()
#         self.decoder_256 = self.create_upsampling_branch_256()
        

#     def create_upsampling_branch_256(self) -> nn.Module:
#         """为256通道输入创建上采样分支"""
        
#         # Skip connections - 处理256通道输入
#         decoder0_skip = nn.Sequential(
#             Conv2DBlock(3, 32, 3, self.drop_rate),
#             Conv2DBlock(32, 64, 3, self.drop_rate),
#         )
#         decoder1_skip = nn.Sequential(
#             Deconv2DBlock(256, 256, dropout=self.drop_rate),
#             Deconv2DBlock(256, 128, dropout=self.drop_rate),
#             Deconv2DBlock(128, 128, dropout=self.drop_rate),
#         )
#         decoder2_skip = nn.Sequential(
#             Deconv2DBlock(256, 256, dropout=self.drop_rate),
#             Deconv2DBlock(256, 256, dropout=self.drop_rate),
#         )
#         decoder3_skip = nn.Sequential(
#             Deconv2DBlock(256, 312, dropout=self.drop_rate)  # 保持和原来一样的bottleneck_dim
#         )

#         # Upsampling - 使用ConvTranspose2d
#         bottleneck_upsampler = nn.ConvTranspose2d(
#             in_channels=256,
#             out_channels=312,
#             kernel_size=2,
#             stride=2,
#             padding=0,
#             output_padding=0,
#         )
#         decoder3_upsampler = nn.Sequential(
#             Conv2DBlock(312 * 2, 312, dropout=self.drop_rate),
#             Conv2DBlock(312, 312, dropout=self.drop_rate),
#             Conv2DBlock(312, 312, dropout=self.drop_rate),
#             nn.ConvTranspose2d(
#                 in_channels=312,
#                 out_channels=256,
#                 kernel_size=2,
#                 stride=2,
#                 padding=0,
#                 output_padding=0,
#             ),
#         )
#         decoder2_upsampler = nn.Sequential(
#             Conv2DBlock(256 * 2, 256, dropout=self.drop_rate),
#             Conv2DBlock(256, 256, dropout=self.drop_rate),
#             nn.ConvTranspose2d(
#                 in_channels=256,
#                 out_channels=128,
#                 kernel_size=2,
#                 stride=2,
#                 padding=0,
#                 output_padding=0,
#             ),
#         )
#         decoder1_upsampler = nn.Sequential(
#             Conv2DBlock(128 * 2, 128, dropout=self.drop_rate),
#             Conv2DBlock(128, 128, dropout=self.drop_rate),
#             nn.ConvTranspose2d(
#                 in_channels=128,
#                 out_channels=64,
#                 kernel_size=2,
#                 stride=2,
#                 padding=0,
#                 output_padding=0,
#             ),
#         )
#         decoder0_header = nn.Sequential(
#             Conv2DBlock(64 * 2, 64, dropout=self.drop_rate),
#             Conv2DBlock(64, 64, dropout=self.drop_rate),
#         )

#         decoder = nn.Sequential(
#             OrderedDict(
#                 [
#                     ("decoder0_skip", decoder0_skip),
#                     ("decoder1_skip", decoder1_skip),
#                     ("decoder2_skip", decoder2_skip),
#                     ("decoder3_skip", decoder3_skip),
#                     ("bottleneck_upsampler", bottleneck_upsampler),
#                     ("decoder3_upsampler", decoder3_upsampler),
#                     ("decoder2_upsampler", decoder2_upsampler),
#                     ("decoder1_upsampler", decoder1_upsampler),
#                     ("decoder0_header", decoder0_header),
#                 ]
#             )
#         )

#         return decoder




#     def create_sam_prompt_encoder(self, sam_checkpoint_path=None):
#         """创建SAM的prompt encoder并加载预训练权重"""
#         from segment_anything.modeling.prompt_encoder import PromptEncoder
        
#         # SAM-ViT-H的配置
#         prompt_encoder = PromptEncoder(
#             embed_dim=256,  # prompt embedding维度
#             image_embedding_size=(16, 16),  # 与你的image embedding尺寸匹配
#             input_image_size=(256, 256),  # SAM原始输入尺寸
#             mask_in_chans=16,  # mask提示的输入通道数
#         )
        
#         # ✅ 从checkpoint加载预训练权重
#         if sam_checkpoint_path is not None and Path(sam_checkpoint_path).exists():
#             try:
#                 sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path)
#                 prompt_encoder.load_state_dict(sam.prompt_encoder.state_dict())
#                 print("✅ Successfully loaded SAM prompt encoder weights")
#             except Exception as e:
#                 print(f"⚠️ Failed to load SAM prompt encoder weights: {e}")
#                 print("Using randomly initialized prompt encoder")
        
#         return prompt_encoder

#     def wrap_dinov3_blocks_with_adapters(self, freeze_backbone=True):
#         # 获取DINOv3 encoder的blocks
#         # print(f"myencoder{dir(self.encoder.named_parameters)}")
#         # print(f"encoder{dir(self.encoder.timm_model)}")

#         if hasattr(self.encoder.timm_model, 'blocks'):
#             # transformers库的标准结构
#             blocks = self.encoder.timm_model.blocks

        
#         adapted_blocks = []
#         for block in blocks:
#             adapted_block = DINOv3Adapter2(block)  # 创建Adapter包装器
#             adapted_blocks.append(adapted_block)
        
#         # print(f"lengh of apapted_blocks{len(adapted_blocks)}")

#         if hasattr(self.encoder.timm_model, 'blocks'):
#             self.encoder.timm_model.blocks = nn.ModuleList(adapted_blocks)
        

#         if freeze_backbone:
#             for name, param in self.encoder.timm_model.named_parameters():
#                 if 'prompt_learn' not in name and 'adapter_mlp' not in name:  # 只训练Adapter参数
#                     param.requires_grad = False


#     def create_sam_decoder(self, sam_checkpoint=None):
#         """创建SAM官方解码器"""
#         from segment_anything.modeling.mask_decoder import MaskDecoder
#         from segment_anything.modeling.transformer import TwoWayTransformer

#         offset_branches = 2 if self.regression_loss else 0
    
#         # 计算所有任务需要的总通道数
#         total_channels = (2 + offset_branches) + 2 + self.num_nuclei_classes

#         # SAM解码器参数（基于SAM-B的配置）
#         decoder_config = {
#             'transformer_dim': 256,
#             'transformer': {
#                 'depth': 2,
#                 'embedding_dim': 256,
#                 'mlp_dim': 2048,
#                 'num_heads': 8,
#             },
#             'num_multimask_outputs': total_channels,
#             'iou_head_depth': 3,
#             'iou_head_hidden_dim': 256,
#             'activation': nn.GELU,
#         }
        
#         # 创建SAM解码器
#         mask_decoder = MaskDecoder(
#             transformer_dim=decoder_config['transformer_dim'],
#             transformer=TwoWayTransformer(
#                 depth=decoder_config['transformer']['depth'],
#                 embedding_dim=decoder_config['transformer']['embedding_dim'],
#                 mlp_dim=decoder_config['transformer']['mlp_dim'],
#                 num_heads=decoder_config['transformer']['num_heads'],
#             ),
#             num_multimask_outputs=decoder_config['num_multimask_outputs'],
#             activation=decoder_config['activation'],
#             iou_head_depth=decoder_config['iou_head_depth'],
#             iou_head_hidden_dim=decoder_config['iou_head_hidden_dim'],
#         )
        
#         # 如果提供了SAM checkpoint，加载解码器权重
#         if sam_checkpoint is not None:
#             self.load_sam_decoder_weights(mask_decoder, sam_checkpoint)
        
#         return mask_decoder
    
#     def load_pretrained_encoder(self, model_path: str):
#         """加载预训练的DINOv3编码器"""
#         pass

#     def load_sam_decoder_weights(self, mask_decoder, sam_checkpoint):
#         """从SAM checkpoint加载解码器权重"""
#         from segment_anything import sam_model_registry
#         try:
#             # 加载整个SAM模型
#             sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
#             # 只复制解码器的权重
#             mask_decoder.load_state_dict(sam.mask_decoder.state_dict())
#             print("Successfully loaded SAM decoder weights")
#         except Exception as e:
#             print(f"Failed to load SAM decoder weights: {e}")
#             print("Using randomly initialized SAM decoder")


#     def create_dinov3_encoder(self):
#         """创建DINOv3编码器"""
#         from transformers import AutoModel
        

        
#         backbone_model = AutoModel.from_pretrained(
#             "backbone",
#             torch_dtype=torch.float32,
#             local_files_only=True
#         )
#         return backbone_model



#     def create_empty_prompts(self, batch_size, device):
#         """
#         使用SAM prompt encoder生成空提示embeddings
        
#         Returns:
#             sparse_embeddings: [B, N, 256] 稀疏提示embeddings
#             dense_embeddings: [B, 256, 64, 64] 密集mask embeddings
#         """
#         # ✅ 使用prompt encoder生成空提示（不提供任何点、框或mask）
#         sparse_embeddings, dense_embeddings = self.prompt_encoder(
#             points=None,  # 没有点提示
#             boxes=None,   # 没有框提示
#             masks=None,   # 没有mask提示
#         )
        
#         # 调整batch size（prompt encoder默认返回batch_size=1）
#         if sparse_embeddings.shape[0] == 1 and batch_size > 1:
#             sparse_embeddings = sparse_embeddings.repeat(batch_size, 1, 1)
        
#         if dense_embeddings.shape[0] == 1 and batch_size > 1:
#             dense_embeddings = dense_embeddings.repeat(batch_size, 1, 1, 1)
        
#         return sparse_embeddings.to(device), dense_embeddings.to(device)

#     def forward(self, x: torch.Tensor, retrieve_tokens: bool = False) -> dict:
#         """使用SAM解码器的前向传播"""
#         out_dict = {}
#         B = x.shape[0]
#         device = "cuda:1"
#         # mlp = 0 # 是否使用MLP连接层的标志，0表示不使用，1表示使用

#         # DINOv3编码器前向传播
#         # print(f"x:{x.shape}")
#         outputs = self.encoder(x, output_hidden_states=True)
#         hidden_states = outputs.hidden_states


#         # ✅ 获取多层特征
#         z1 = hidden_states[self.extract_layers[0]]  # 浅层特征
#         z2 = hidden_states[self.extract_layers[1]]  # 中层特征
#         z3 = hidden_states[self.extract_layers[2]]  # 深层特征
#         z4 = hidden_states[self.extract_layers[3]]  # 最深层特征
        
#         # ✅ 特征融合（可以用简单的加权平均或注意力机制）
#         # 方式1: 简单加权
#         # weights = torch.softmax(self.layer_weights, dim=0)
#         # image_embeddings = (
#         #     weights[0] * z1 + 
#         #     weights[1] * z2 + 
#         #     weights[2] * z3 + 
#         #     weights[3] * z4
#         # )
        
        
#         # 获取分类特征
#         classifier_logits = outputs.pooler_output
#         out_dict["tissue_types"] = self.classifier_head(classifier_logits)


#         # print(f"image_embedding{image_embeddings.shape}")
#         image_embeddings = self.projection(z4)  # 投影到256通道
#         z1 = self.projection(z1)  # 投影到256通道
#         z2 = self.projection(z2)  # 投影到256通道
#         z3 = self.projection(z3)  # 投影到256通道
#         # print(f"image_embedding2:{image_embeddings.shape}")

        
#         # 创建虚拟提示
#         sparse_embeddings, dense_embeddings = self.create_empty_prompts(B, device)

#         sparse_embeddings = sparse_embeddings.to(device)
#         image_embeddings = image_embeddings.to(device)
        


#         if self._image_pe_cache is None or self._image_pe_cache.shape[0] != B:
#             self._image_pe_cache = self.pe_layer((16, 16)).unsqueeze(0).repeat(B, 1, 1, 1)
#             self._image_pe_cache = self._image_pe_cache.to(device)
        
#         image_pe = self._image_pe_cache





#         # SAM解码器前向传播
#         # print(f"image_embedding:{image_embeddings.shape}")
#         masks, iou_predictions = self.sam_decoder(
#             image_embeddings=image_embeddings,
#             image_pe=image_pe,
#             sparse_prompt_embeddings=sparse_embeddings,
#             dense_prompt_embeddings=dense_embeddings,
#             multimask_output=True,
#         )
#         masks = masks.to("cuda:1")  # 将 masks 移动到与输出投影层相同的设备
        


#         offset_branches = 2 if self.regression_loss else 0
    
#         # 定义每个任务的通道范围
#         self.channel_splits = {
#             'binary_start': 0,
#             'binary_end': 2 + offset_branches,
#             'hv_start': 2 + offset_branches,
#             'hv_end': 4 + offset_branches,
#             'type_start': 4 + offset_branches,
#             'type_end': 4 + offset_branches + self.num_nuclei_classes
#         }

#         # print(f"masks:{masks.shape}")

#         # print(f"z1:{z1.shape}")
#         # print(f"z2:{z2.shape}")
#         # print(f"z3:{z3.shape}")

#         # 使用共享解码器进行上采样
#         upsampled = self._forward_upsample(x, z1, z2, z3, masks, self.decoder_256)
        
#         # print(f"Debug: 上采样后形状: {upsampled.shape}")
        
#         # 各个任务头
#         if self.regression_loss:
#             nb_map = self.nuclei_binary_map_decoder(upsampled)
#             # print(f"nb_map:{nb_map.shape}")
#             out_dict["nuclei_binary_map"] = nb_map[:, :2, :, :]
#             out_dict["regression_map"] = nb_map[:, 2:, :, :]
#         else:
#             out_dict["nuclei_binary_map"] = self.nuclei_binary_map_decoder(upsampled)
        
#         out_dict["hv_map"] = self.hv_map_decoder(upsampled)
#         out_dict["nuclei_type_map"] = self.nuclei_type_maps_decoder(upsampled)

#         if retrieve_tokens:
#             # 使用最后一层的空间特征作为tokens
#             out_dict["tokens"] = z4


#         # a = out_dict["nuclei_binary_map"]
#         # print(f"out_dict:{a.shape}")
#         return out_dict

#     def init_dinov3_s(self):
#         self.embed_dim = 384
#         self.depth = 12
#         self.num_heads = 6
#         self.extract_layers = [2, 5, 8, 11]
#         self.patch_size = 16

#     def init_dinov3_b(self):
#         self.embed_dim = 768
#         self.depth = 12
#         self.num_heads = 12
#         self.extract_layers = [2, 5, 8, 11]
#         self.patch_size = 14

#     def init_dinov3_l(self):
#         self.embed_dim = 1024
#         self.depth = 24
#         self.num_heads = 16
#         self.extract_layers = [6, 12, 18, 24]
#         self.patch_size = 14

#     def init_dinov3_g(self):
#         self.embed_dim = 1536
#         self.depth = 40
#         self.num_heads = 24
#         self.extract_layers = [10, 20, 30, 40]
#         self.patch_size = 14


# class DINOv3Adapter2(nn.Module):
#     def __init__(self, blk):
#         super().__init__()
#         self.block = blk
#         dim = blk.attn.qkv.in_features
#         reduction = 16
#         self.prompt_learn = nn.Sequential(
#             nn.Linear(dim, dim // reduction),
#             nn.GELU(),
#             nn.Linear(dim // reduction, dim),
#             nn.GELU()
#         )
    
#     def forward(self, x, rope=None):
#         prompt = self.prompt_learn(x)
#         prompted = x + prompt
#         return self.block(prompted, rope=rope)


# class DINOv3Adapter(nn.Module):
#     """DINOv3的Adapter包装器，使用archs.py中的shiftedBlock"""
    
#     def __init__(self, block):
#         super().__init__()
#         self.block = block
        
#         # 获取block的输入维度
#         if hasattr(block, 'attention') and hasattr(block.attention, 'qkv'):
#             dim = block.attention.qkv.in_features
#         elif hasattr(block, 'attn') and hasattr(block.attn, 'qkv'):
#             dim = block.attn.qkv.in_features
#         else:
#             dim = 768
        
#         self.dim = dim
#         reduction = 16
        
#         # Adapter网络结构
#         self.prompt_learn = nn.Sequential(
#             nn.Linear(dim, dim // reduction),
#             nn.GELU(),
#             nn.Linear(dim // reduction, dim),
#             nn.GELU()
#         )
        
#         mlp_ratio = 4
#         self.adapter_block = shiftedBlock(
#             dim=dim,
#             num_heads=4, 
#             mlp_ratio=mlp_ratio,
#             drop=0.1,
#             attn_drop=0.1,
#             drop_path=0.1,
#             act_layer=nn.GELU,
#             norm_layer=nn.LayerNorm
#         )
    
#     def forward(self, x, rope=None):
#         # 保存原始输入
#         original_x = x

#         image_token = x[:,:256,:]
#         special_token = x[:,256:,:]
        
#         # 应用prompt learning
#         prompt = self.prompt_learn(image_token)
#         prompted = image_token + prompt

        
#         # 🔥 应用shiftedBlock（需要H, W参数）
#         B, N, C = prompted.shape
#         # H = W = int(math.sqrt(N))  # 假设是正方形特征图
#         H = W = 16
#         # print(f"B:{B} N: {N} C: {C}")
#         # print(f"orignal x:{x.shape}")
#         adapted_output = self.adapter_block(prompted, H, W)
        
#         # 组合输出：原始输入 + Adapter输出
#         image_token_output = image_token + 0.5 * adapted_output
        
#         final_output = torch.cat([image_token_output, special_token], dim = 1)

#         # 通过原始block
#         if rope is not None:
#             return self.block(final_output, rope=rope)
#         else:
#             return self.block(final_output)