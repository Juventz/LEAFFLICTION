# 42_leaffliction

Computer vision project for analyzing, augmenting, transforming, and classifying
leaf disease images.

## Setup

This project uses `pyproject.toml` and `uv.lock` for dependencies.

Create or update the virtual environment:

```bash
uv sync
```

Activate it:

```bash
source .venv/bin/activate
```

You can then run the scripts with Python:

```bash
python srcs/Distribution.py ./Apple
```

Or without activating the environment:

```bash
uv run python srcs/Distribution.py ./Apple
```

## Part 1 - Distribution

Script:

```bash
python srcs/Distribution.py ./Apple
```

Goal:

- browse a directory containing class subdirectories;
- count the images in each class;
- display a pie chart and a bar chart;
- use directory names automatically to label the charts.

Examples:

```bash
python srcs/Distribution.py ./Apple
python srcs/Distribution.py ./Grape
python srcs/Distribution.py ./images
```

The script is not hard-coded for `Apple`; it works with any dataset directory
that follows the same structure: one root directory containing class
subdirectories.

## Part 2 - Data Augmentation

Script:

```bash
python srcs/Augmentation.py "./Apple/Apple_healthy/image (1).JPG"
```

Single-image mode:

- displays 6 augmentations;
- saves the 6 augmented images in the same directory as the source image;
- names each output file with the original filename followed by the
  augmentation type.

Augmentations used:

- `Flip`
- `Rotate`
- `Scaling`
- `Illumination`
- `Contrast`
- `Projective`

Example output files:

```text
image (1)_Flip.JPG
image (1)_Rotate.JPG
image (1)_Scaling.JPG
image (1)_Illumination.JPG
image (1)_Contrast.JPG
image (1)_Projective.JPG
```

Directory mode:

```bash
python srcs/Augmentation.py ./Apple
python srcs/Augmentation.py ./Grape
```

This copies the dataset into `augmented_directory/<dataset_name>` and balances
the classes by adding augmented images until every class has the same number of
images.

## Part 3 - Image Transformation

Script:

```bash
python srcs/Transformation.py "./Apple/Apple_healthy/image (1).JPG"
```

Single-image mode:

- displays the original image;
- displays at least 6 feature-extraction transformations.

Transformations:

- `GaussianBlur`
- `Mask`
- `RoiObjects`
- `AnalyzeObject`
- `Pseudolandmarks`
- `ColorHistogram`

Directory mode:

```bash
python srcs/Transformation.py -src ./Apple/Apple_healthy -dst ./dst_directory -mask
```

This processes every image in the source directory and saves the
transformations in the destination directory.

Example output files:

```text
image (1)_GaussianBlur.JPG
image (1)_Mask.JPG
image (1)_RoiObjects.JPG
image (1)_AnalyzeObject.JPG
image (1)_Pseudolandmarks.JPG
image (1)_ColorHistogram.JPG
```

Show the available options:

```bash
python srcs/Transformation.py -h
```

## Part 4 - Classification

Train the model:

```bash
python srcs/train.py ./Apple
```

Link Kaggle:  https://www.kaggle.com/code/akadilkalimoldayev/42-leaffliction/notebook
Signature  : sha256sum model_artifacts.zip 


The training script:

- splits the dataset into training and validation sets;
- balances the training files internally;
- trains an EfficientNetB0 classifier;
- saves `best_model.keras`, `class_names.txt`, and `model_artifacts.zip`;
- prints the validation accuracy at the end.

Run a prediction:

```bash
python srcs/predict.py . ./Apple/Apple_healthy/image.JPG
```

The first argument is the directory containing `best_model.keras` and
`class_names.txt`. The second argument is the image to classify.

## Turn-In

Only the programs and `signature.txt` should be present in the Git repository.
Do not commit the dataset, augmented dataset, model files, or zip archive.

Create the signature from the final zip archive:

```bash
shasum model_artifacts.zip > signature.txt
```

Before submitting, remove `leaffliction.subject.pdf` from the repository and
make sure `signature.txt` matches the zip that contains the dataset/model used
for evaluation.

## Quick Check

Check that all scripts compile:

```bash
python -m py_compile srcs/Distribution.py srcs/Augmentation.py srcs/Transformation.py srcs/train.py srcs/predict.py srcs/image_utils.py
```

Check the Python norm with `flake8`:

```bash
python -m flake8 srcs
```
