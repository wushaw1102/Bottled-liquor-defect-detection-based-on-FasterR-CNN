# Annotation files and image layout

```
data/
├── annotations/
│   ├── instances_all.json      all 4,516 images / 6,945 boxes (COCO format)
│   ├── instances_train.json    3,161 images / 4,868 boxes   (70 %)
│   ├── instances_val.json        903 images / 1,405 boxes   (20 %)
│   └── instances_test.json       452 images /   672 boxes   (10 %)
├── sample/
│   ├── images/                 60 photos committed to this repository (4 MB)
│   └── annotations.json        the subset of instances_test.json for those 60
└── images/                     <-- put the full image set here (not committed)
```

## Why the full images are not in the repository

The complete dataset is 4,516 JPEGs at 658 × 492 px, roughly **260 MB**. Git is a
poor host for that (every clone downloads the whole history), so only a 60-image,
class-balanced sample is committed. Everything else - annotations, splits, code -
is complete and versioned.

To run the full pipeline, copy the original photos into `data/images/`:

```bash
# the file names must match the "file_name" fields in the annotation files
ls data/images | head -3
# img_0000004.jpg  img_0000012.jpg  img_0000014.jpg
```

Then verify the pairing before training:

```bash
python scripts/prepare_dataset.py \
    --annotation-file data/annotations/instances_all.json \
    --image-dir data/images --verify-only
```

The command prints the per-class histogram per split and flags any missing image
or orphan annotation.

## COCO format

Each annotation file is a standard COCO dictionary:

```json
{
  "images": [
    {"id": 1, "file_name": "img_0017151.jpg", "width": 658, "height": 492}
  ],
  "annotations": [
    {"id": 213, "image_id": 1, "category_id": 2,
     "bbox": [165.14, 53.71, 39.86, 63.29], "area": 2522.7, "iscrowd": 0}
  ],
  "categories": [
    {"id": 1, "name": "瓶盖破损"}, ...
  ]
}
```

`bbox` is `[x_min, y_min, width, height]` in pixels, as in the COCO spec.

`category_id = 0` is named `背景` ("background") in the original annotations and
has 1,170 boxes. torchvision detectors reserve label 0 for the background slot,
so those boxes can never be learned. They are kept in the files for fidelity with
the original data and excluded from every reported metric; pass
`--drop-background` to `scripts/prepare_dataset.py` if you train from scratch and
want them removed from the loss entirely.

## Rebuilding the splits

```bash
python scripts/prepare_dataset.py \
    --annotation-file data/annotations/instances_all.json \
    --image-dir data/images \
    --output-dir data/annotations \
    --train 0.7 --val 0.2 --seed 42
```

## Rebuilding the committed sample

```bash
python scripts/make_sample.py --count 60 --per-class 4
```
