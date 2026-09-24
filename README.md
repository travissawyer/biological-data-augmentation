# Cross-species image translation for biological data augmentation

Reference implementation of the two models used in *Unpaired Animal-to-Human
Translation of Label-Free Multiphoton Microscopy Images: A Demonstration of
Cross-Species Biological Augmentation in Pancreatic Neuroendocrine Tumors*.

Two parts, each usable on its own:

| Folder | What it is |
|---|---|
| `translation/` | Class-conditioned unpaired image translation between two domains |
| `classifier/` | VGG16 binary classifier for the downstream task |

---

## What data you need

**Animal images** — the source domain. Images of the tissue of interest acquired
from an animal model, each with a class label (normal or tumor).

**Human images** — the target domain. Images acquired from human tissue with the
same imaging modality and channel configuration, each with a class label.

Both domains should be acquired the same way. The translation is unpaired, so
there is no need for any correspondence between individual animal and human
images, and the two sets do not need to be the same size.

> **Use separate groups of human specimens for the two models.** The human
> images used to train the translation model should not overlap with the human
> images used to train or evaluate the classifier. If the same specimens appear
> in both, the translator has seen the classifier's held-out data, and the
> downstream result is no longer a clean measure of what the translated images
> contribute.

---

## Install

```
pip install -r requirements.txt
```

---

## 1. Translation model

### Input

A CSV listing the images for both domains:

| column | meaning |
|---|---|
| `path` | path to the image, absolute or relative to `--data_root` |
| `domain` | `0` for animal (source), `1` for human (target) |
| `label` | class id, `0` or `1` |

```csv
path,domain,label
animal/img_0001.png,0,0
animal/img_0002.png,0,1
human/img_0001.png,1,0
```

Only the human specimens reserved for translation belong in this file.

### Train

```
python -m translation.train --csv_path labels_translation.csv --data_root /path/to/images --save_dir ckpts
```

For the ablation in which the class label reaches neither the decoder nor the discriminator:

```
python -m translation.train --csv_path labels_translation.csv --save_dir ckpts_uncond --no_class_cond
```

Checkpoints are written to `--save_dir` as `best.pt` (lowest source-domain
reconstruction error on the held-out split) and `last.pt`, alongside the full
configuration.

### Generate translated images

```
python -m translation.translate --ckpt ckpts/best.pt --csv_path labels_translation.csv --out_dir translated
```

Writes one translated image per animal input, plus `manifest.csv` recording the
style mode and conditioning setting each image was produced with. `--style_mode
random` draws an independent style code per image; `--style_mode zero` fixes the
style code to zero and produces one deterministic image per input.

### Architecture

Content/style disentanglement with AdaIN decoding, following MUNIT and DRIT; a
KL-regularized Gaussian style latent; cycle consistency; and an auxiliary class
head on the discriminator, following AC-GAN.

The one departure from standard MUNIT is where the class label enters. In MUNIT
the AdaIN affine parameters are a function of the style code alone. Here the
label is embedded and concatenated to the style code before the parameter MLP,
so the same content and the same style code produce different normalization
parameters for each class. The auxiliary class head is applied to translated
images against the *source* label, which is what carries the class across the
domain boundary.

---

## 2. Classifier

### Input

A CSV listing the images and how they are split:

| column | meaning |
|---|---|
| `path` | path to the image |
| `label` | class id, `0` or `1` |
| `split` | `train`, `val` or `test` |
| `source_id` | optional — specimen identifier, carried through to the predictions |
| `origin` | optional — free-text tag, e.g. `real` or `translated`. Recorded, never trained on |

```csv
path,label,split,source_id,origin
human/img_0001.png,0,train,S001,real
translated/normal/img_0001_translated.png,0,train,A014,translated
human/img_0044.png,1,test,S052,real
```

### Train

```
python -m classifier.train --manifest manifest_baseline.csv --data_root /path/to/images --out_dir runs/baseline
python -m classifier.train --manifest manifest_translated.csv --data_root /path/to/images --out_dir runs/translated
```

Useful options:

| Option | Purpose |
|---|---|
| `--preprocess unit \| imagenet` | `unit` scales to [0, 1]; `imagenet` additionally applies the channel mean/std the pretrained weights were trained with |
| `--init pretrained \| scratch` | ImageNet-pretrained backbone, or random initialization |
| `--reg_strength` | Strength of the explicit L2 penalty. A value suited to a pretrained backbone is often far too strong from a random start |

Each run writes `test_predictions.csv` with one row per test image, including
`source_id`, so predictions can be pooled per specimen as well as per image.

### Architecture

VGG16 convolutional stack, frozen except the convolutions among the last N
Keras-equivalent layers, with dropout after each of those layers. Head is
GlobalMaxPool → Dense(1024, ReLU) → Dropout → Dense(1024, ReLU) → Dropout →
Dense(1, sigmoid). Loss is binary cross-entropy plus an explicit L2 penalty on
the trainable convolution weights and all three head weight matrices.

---
## Citation

Please cite the paper if you use this code.
