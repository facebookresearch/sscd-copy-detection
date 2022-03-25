# Reproduce evaluation results

## DISC evaluations

On an 8-GPU server, run the following to evaluate the `sscd_disc_mixup` model
using our default preprocessing (resize the short edge to 288, preserving
the aspect ratio).

```bash
sscd/disc_eval.py --disc_path /path/to/disc2021 --gpus=8 \
  --output_path=/path/to/eval/output \
  --size=288 --preserve_aspect_ratio=true \
  --backbone=CV_RESNET50 --dims=512 --model_state=/path/to/sscd_disc_mixup.classy.pt
```

After ~2 hours (on an 8 GPU machine), this command produces a CSV file,
`disc_metrics.csv`, in the configured `--output_path`:

```
codec,score_norm,uAP,accuracy-at-1,recall-at-p90
Flat,None,0.6093859526781344,0.7757,0.3766
"PCAW512,L2norm,Flat",None,0.6142708346057645,0.782,0.3825
Flat,"1.00[0,2]",0.7180449145415637,0.7757,0.6251
"PCAW512,L2norm,Flat","1.00[0,2]",0.7242754343531326,0.782,0.6308
```

The columns of this file are:
* `codec`: the descriptor postprocessing codec. Our paper results use
  `PCAW<D>,L2norm,Flat`, where `<D>` is the descriptor dimensionality.
* `score_norm`: we evaluate with and without score normalization. The
  three numbers are: `<weight>[first, last]`, where weight is &beta;
  in equation 8, `first` and `last` are zero-based indices of the first
  and last neighbors to include (inclusive).

The two `PCAW512,L2norm,Flat` rows correspond to &micro;AP 61.4%
and &micro;APSN (with score normalization) of 72.4%, similar to the
"SSCD DISC adv.+mixup" row in Table 2 (61.5% and 72.5%, respectively).

Most of our models are 512d ResNet50 models, and can be evaluated by changing just
the `--model_state` argument.
To evaluate the `sscd_large` model, use: `--backbone=CV_RESNEXT101 --dims=1024`

Note that our results use the DISC2021 validation set, as the test set was not
yet available when the paper was written.

## Copydays + 10k distractors (CD10K) evaluations

We evaluate using 10K distractors and a 20K whitening set from YFCC100M.

To evaluate the `sscd_disc_mixup` model using default settings, run:

```bash
sscd/copydays_eval.py --gpus=8 --copydays_path /path/to/copydays \
  --distractor_path /path/to/distractors \
  --codec_train_path /path/to/whitening \
  --output_path=/path/to/eval/output \
  --backbone=CV_RESNET50 --dims=512 \
  --model_state=/path/to/sscd_disc_mixup.classy.pt \
  --size=288 --preserve_aspect_ratio=true \
  --codecs="PCAW512,L2norm,Flat;PCA512,L2norm,Flat"
```

The command above produces evaluation metrics with two embedding
postprocessings: with whitening (`PCAW512,L2norm,Flat`, as reported
in the paper), and with simple centering (`PCA512,L2norm,Flat`).

After ~10 minutes (on an 8 GPU machine), this command will produce a
CSV file, `copydays_metrics.csv`, within the configured `--output_path`:

```
codec,strong_mAP,overall_uAP
"PCAW512,L2norm,Flat",0.865810403059819,0.9823856615515755
"PCA512,L2norm,Flat",0.8843126681663799,0.9820800446423634
```

The `strong_mAP` column shows mAP on the strong subset of Copydays,
and `overall_uAP` contains &micro;AP over the full dataset.
This shows that, with whitening, we get 86.58% mAP and 98.24% &micro;AP,
similar to the 86.6 mAP and 98.1 &micro;AP values reported in the
SSCD row in Table 3.

Note that SSCD has better strong subset mAP metrics (although often
reduced &micro;AP) when evaluated on CD10K without whitening,
using simple centering and L2 normalization (the `PCA512,L2norm,Flat`
row in the CSV): 88.4% here versus 86.5%.

We explore image preprocessing methods used by various baselines on Copydays,
such as resizing to square tensors (`--preserve_aspect_ratio=false`)
and resizing the long edge to a specified size
(`--size=800 --resize_long_edge=true --preserve_aspect_ratio=false`).

## FAISS codec strings

The evaluations above specify descriptor postprocessing methods
(eg. whitening and L2 normalization) using
[FAISS codec strings](https://github.com/facebookresearch/faiss/wiki/The-index-factory).

For instance, if we start with 512d L2 normalized descriptors
(as our ResNet50 models produce), and use the codec string
`PCAW512,L2norm,Flat`,
this means that descriptors will be whitened, then L2 normalized again,
before being used for retrieval.

Evaluation scripts can evaluate features using multiple codecs separated
by `;`, such as `PCAW512,L2norm,Flat;PCA512,L2norm,Flat` to evaluate
both whitening followed by L2 normalization and centering followed by
L2 normalization.

(`PCA512` centers a 512 dimensional representation by subtracting the mean
before applying an orthogonal PCA projection. That projection does not change
descriptor distances, so the only distance-changing effect is the centering.)
