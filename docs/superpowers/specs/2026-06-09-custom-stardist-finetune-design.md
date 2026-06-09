# Custom StarDist Fine-Tuning Notebook Design

## Goal

Create a second notebook for fine-tuning the existing StarDist2D model trained by `stardist_dsb2018_train.ipynb`.

The notebook will train on a small custom dataset, expected to contain at most about 50 images, while preserving the existing base model weights.

## Files and Directories

- New notebook: `stardist_custom_finetune.ipynb`
- New data directory: `custom_data/`
- Base model directory: `models/stardist_dsb2018_from_scratch/`
- Base weights: `models/stardist_dsb2018_from_scratch/weights_best.h5`
- Fine-tuned output directory pattern: `models/stardist_dsb2018_finetune_custom_YYYYMMDD_HHMMSS/`

The base model path is fixed. The notebook will not automatically select the newest timestamped model.

## Custom Data Layout

`custom_data` sits next to `data-science-bowl-2018`:

```text
stardistTest/
  data-science-bowl-2018/
  custom_data/
    image_name_001/
      images/
        image_name_001.png
      masks/
        mask_001.png
        mask_002.png
    image_name_002/
      images/
        image_name_002.png
      masks/
        mask_001.png
```

Each sample folder must contain:

- `images/*.png`: one source image
- `masks/*.png`: one binary PNG mask per instance

This matches the `stage1_train` layout used by Data Science Bowl 2018.

## Notebook Flow

1. Import packages and configure CUDA/CPU device using the same pattern as the existing training notebook.
2. Locate the project root and ensure `custom_data/` exists.
3. Validate custom samples:
   - include only folders with at least one image and at least one mask
   - warn about skipped folders
   - stop with a clear error if fewer than 2 usable samples exist
4. Load images and merge per-instance masks into integer instance label images.
5. Standardize image and mask dimensions to `1536 x 1024` pixels, interpreted as width x height.
6. Normalize images with percentile normalization.
7. Split data deterministically into train and validation sets, keeping at least one validation image.
8. Create a StarDist2D model with the same architecture as the base model.
9. Load fixed base weights from `models/stardist_dsb2018_from_scratch/weights_best.h5`.
10. Freeze most layers and keep only the last configurable number of layers trainable.
11. Fine-tune with a low learning rate and augmentation.
12. Optimize thresholds on validation data when validation data exists.
13. Show a validation prediction sample.

## Dimension Standardization

Custom images are expected to be `1536 x 1024` pixels. In code this is represented as `(height=1024, width=1536)`.

The notebook will define:

- `TARGET_WIDTH = 1536`
- `TARGET_HEIGHT = 1024`
- `TARGET_SHAPE = (TARGET_HEIGHT, TARGET_WIDTH)`

If an image already has this shape, preprocessing leaves it unchanged.

If an image or mask has a different shape:

- source images are resized to `TARGET_SHAPE` with bilinear interpolation and anti-aliasing
- binary instance masks are resized to `TARGET_SHAPE` with nearest-neighbor interpolation
- resized masks are thresholded back to boolean masks before merging into instance labels

Nearest-neighbor resizing is required for masks so that instance boundaries do not become fractional labels.

## Fine-Tuning Defaults

Default values are conservative because the custom dataset is small:

- `PATCH_SIZE = (256, 256)`
- `BATCH_SIZE = 1`
- `EPOCHS = 30`
- `LEARNING_RATE = 1e-5`
- `VALIDATION_SPLIT = 0.2`
- `UNFREEZE_LAST_N_LAYERS = 8`
- `STEPS_PER_EPOCH = max(20, len(X_train) * 8)`

The notebook will expose these values in a settings cell so they can be changed easily.

## Overfitting Controls

The notebook will reduce overfitting risk with:

- low learning rate
- small batch size
- deterministic validation split
- output saved to a new timestamped model directory
- layer freezing by default
- random 90-degree rotations and flips
- mild intensity scale and shift augmentation

If the custom images differ strongly from the DSB images, the user can increase `UNFREEZE_LAST_N_LAYERS` or reduce freezing after an initial run.

## Error Handling

The notebook will fail early with clear messages when:

- the base model directory is missing
- `weights_best.h5` is missing
- no usable custom samples are found
- fewer than 2 usable samples are available
- a mask shape does not match its image shape before preprocessing in a way that cannot be resized

It will create `custom_data/` if missing, then raise an instructional error explaining the expected folder layout.

## Non-Goals

- Do not overwrite `models/stardist_dsb2018_from_scratch/weights_best.h5`.
- Do not train from scratch.
- Do not upload or copy the custom dataset.
- Do not generate Kaggle submissions in this notebook.
- Do not automatically choose timestamped model folders as the base model.

## Verification

Implementation should be checked by:

- confirming `custom_data/` is created
- confirming notebook JSON is valid
- confirming the notebook references the fixed base model path
- confirming preprocessing standardizes images and masks to `(1024, 1536)`
- confirming fine-tuned outputs use a timestamped model name
- optionally running the notebook only after valid custom data is present
