---
tags:
- image-feature-extraction
- timm
- transformers
pipeline_tag: image-feature-extraction
library_name: timm
license: other
license_name: dinov3-license
license_link: https://ai.meta.com/resources/models-and-libraries/dinov3-license
datasets:
- lvd-1689m
---
# Model card for vit_small_patch16_dinov3.lvd1689m

A DINOv3 ViT model image feature encoder. Distilled on LVD-1689M from the DINOv3 ViT-7B model.

## Model Notes
* The original model weights ended up with all QKV projection biases being zeroes. For `timm`, have disabled the QKV bias (`qkv_bias=False`) for the models and not loaded the zero weights. For some model sizes there are variants with `qkvb` in the name that have the bias enabled (`qkv_bias=True`), but zero, to match the behaviour of `transformers` and original models.
* The original models keep RoPE periods as a persistent `bfloat16` buffer. `timm` generates `float32` periods at init. This results in some numerical differences, however the `timm` approach should be less problematic running on devices without bfloat16 support, and appears to work as well if not slightly better for fine-tuning. `model.rope.periods = model.rope.periods.to(torch.bfloat16).to(torch.float32)` will truncate the periods to bfloat16 and result in matching outputs.

## Model Details
- **Model Type:** Image Feature Encoder
- **Model Stats:**
  - Params (M): 21.6
  - GMACs: 6.3
  - Activations (M): 17.0
  - Image size: 256 x 256
- **Original:** https://github.com/facebookresearch/dinov3
- **License:** [DINOv3](https://ai.meta.com/resources/models-and-libraries/dinov3-license)
- **Dataset:** LVD-1689M
- **Papers:**
  - DINOv3: https://arxiv.org/abs/2508.10104
  - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale: https://arxiv.org/abs/2010.11929v2
  - PyTorch Image Models: https://github.com/huggingface/pytorch-image-models

## Model Usage
### Image Classification
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model('vit_small_patch16_dinov3.lvd1689m', pretrained=True)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
```

### Feature Map Extraction
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model(
    'vit_small_patch16_dinov3.lvd1689m',
    pretrained=True,
    features_only=True,
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

for o in output:
    # print shape of each feature map in output
    # e.g.:
    #  torch.Size([1, 384, 16, 16])
    #  torch.Size([1, 384, 16, 16])
    #  torch.Size([1, 384, 16, 16])

    print(o.shape)
```

### Image Embeddings
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model(
    'vit_small_patch16_dinov3.lvd1689m',
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # output is (batch_size, num_features) shaped tensor

# or equivalently (without needing to set num_classes=0)

output = model.forward_features(transforms(img).unsqueeze(0))
# output is unpooled, a (1, 261, 384) shaped tensor

output = model.forward_head(output, pre_logits=True)
# output is a (1, num_features) shaped tensor
```

## Model Comparison
See the associated paper for details on the evaluation protocols

### Results for ViT backbones pretrained (or distilled) on web (LVD-1689M)

| Model | IN-ReaL | IN-R | Obj.Net | Ox.-H | ADE20k | NYU↓ | DAVIS | NAVI | SPair |
|-------|---------|------|---------|-------|--------|------|-------|------|-------|
| **Global Tasks** | | | | | **Dense Tasks** | | | | |
| DINOv3 ViT-S/16 | 87.0 | 60.4 | 50.9 | 49.5 | 47.0 | 0.403 | 72.7 | 56.3 | 50.4 |
| DINOv3 ViT-S+/16 | 88.0 | 68.8 | 54.6 | 50.0 | 48.8 | 0.399 | 75.5 | 57.1 | 55.2 |
| DINOv3 ViT-B/16 | 89.3 | 76.7 | 64.1 | 58.5 | 51.8 | 0.373 | 77.2 | 58.8 | 57.2 |
| DINOv3 ViT-L/16 | 90.2 | 88.1 | 74.8 | 63.1 | 54.9 | 0.352 | 79.9 | 62.3 | 61.3 |
| DINOv3 ViT-H+/16 | 90.3 | 90.0 | 78.6 | 64.5 | 54.8 | 0.352 | 79.3 | 63.3 | 56.3 |
| DINOv3 ViT-7B/16 | 90.4 | 91.1 | 91.1 | 72.8 | 55.9 | 0.309 | 79.7 | 64.4 | 58.7 |

### Results for ConvNeXt backbones distilled on web (LVD-1689M)

| Model | IN-ReaL @256px | IN-ReaL @512px | IN-R @256px | IN-R @512px | Obj.Net @256px | Obj.Net @512px | ADE20k | NYU↓ |
|-------|----------------|----------------|-------------|-------------|----------------|----------------|--------|------|
| **Global Tasks** | | | | | | | **Dense Tasks** | |
| DINOv3 ConvNeXt Tiny | 86.6 | 87.7 | 73.7 | 74.1 | 52.6 | 58.7 | 42.7 | 0.448 |
| DINOv3 ConvNeXt Small | 87.9 | 88.7 | 73.7 | 74.1 | 52.6 | 58.7 | 44.8 | 0.432 |
| DINOv3 ConvNeXt Base | 88.5 | 89.2 | 77.2 | 78.2 | 56.2 | 61.3 | 46.3 | 0.420 |
| DINOv3 ConvNeXt Large | 88.9 | 89.4 | 81.3 | 82.4 | 59.3 | 65.2 | 47.8 | 0.403 |

### Results for ViT backbones pretrained (or distilled) on satellite (SAT-493M)

#### (GEO-Bench) Classification

| Model | m-BEnet | m-brick-kiln | m-eurosat | m-forestnet | m-pv4ger | m-so2sat | mean |
|-------|---------|--------------|-----------|-------------|----------|----------|------|
| DINOv3 ViT-L/16 | 73.0 | 96.5 | 94.1 | 60.6 | 96.0 | 57.4 | 79.6 |
| DINOv3 ViT-7B/16 | 74.0 | 97.2 | 94.8 | 62.3 | 96.1 | 62.1 | 81.1 |

#### (GEO-Bench) Segmentation

| Model | m-cashew | m-chesapeake | m-NeonTree | m-nz-cattle | m-pv4ger-seg | m-SA-crop | mean |
|-------|----------|--------------|------------|-------------|--------------|-----------|------|
| DINOv3 ViT-L/16 | 94.2 | 75.6 | 61.8 | 83.7 | 95.2 | 36.8 | 74.5 |
| DINOv3 ViT-7B/16 | 94.1 | 76.6 | 62.6 | 83.4 | 95.5 | 37.6 | 75.0 |

## Citation
```bibtex
@article{simeoni2025dinov3,
  title={DINOv3},
  author={Sim{'e}oni, Oriane and Vo, Huy V and Seitzer, Maximilian and Baldassarre, Federico and Oquab, Maxime and Jose, Cijo and Khalidov, Vasil and Szafraniec, Marc and Yi, Seungeun and Ramamonjisoa, Micha{"e}l and others},
  journal={arXiv preprint arXiv:2508.10104},
  year={2025}
}
}
```
```bibtex
@article{dosovitskiy2020vit,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={ICLR},
  year={2021}
}
```
```bibtex
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/huggingface/pytorch-image-models}}
}
```
