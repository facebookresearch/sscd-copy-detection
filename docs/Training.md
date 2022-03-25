# Training SSCD models

We use 4 8-GPU nodes to train SSCD models.
We run `sscd/train.py` once on each training machine, passing
[PyTorch distributed](https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization)
environemnt variables to each command to coordinate workers.

The train command on each worker is as follows:
```
MASTER_ADDR="<first worker hostname>" MASTER_PORT="20285" NODE_RANK="<this workers rank, 0-3>" WORLD_SIZE=32 \
  ./sscd/train.py --nodes="<all worker hostnames>" --gpus=8 \
  --train_dataset_path=/path/to/disc/training \
  --entropy_weight=30 --augmentations=ADVANCED --mixup=true \
  --output_path=/path/to/train/output
```

## Training using Slurm

We orchestrate this using [Slurm](https://slurm.schedmd.com/documentation.html),
and provide a wrapper script that translates Slurm environment variables to
PyTorch distributed environment variables.
(The next release of PyTorch Lightning should detect environment variables from Slurm and other cluster
environments automatically.)

```
srun --nodes=4 --gpus-per-node=8 -mem=0 \
  --cpus-per-task=<num CPUs> --ntasks-per-node=1 \
  ./bin/train_slurm_wrapper.sh --train_dataset_path=/path/to/disc/training \
  --entropy_weight=30 --augmentations=ADVANCED --mixup=true \
  --output_path=/path/to/train/output
```

### Evaluating models trained using this codebase

Training produces a checkpoint file within the provided `--output_path`,
for instance at `<provided path>/lightning_logs/version_<version id>/checkpoints/epoch=99-step=24399.ckpt`,
where `<version id>` is an integer ID chosen by the Lightning framework.

Our evaluation commands can load model settings and weights from
these checkpoints via the `--checkpoint=` parameter.
When using `--checkpoint=`, omit other model parameters
(i.e. don't set `--backbone`, `--dims` or `--model_state`).

## Advice for extending SSCD

To extend SSCD, for instance using different trunks,
batch size, image augmentations, or optimizers, it may be necessary
to reduce the entropy weight (&lambda;, via the `--entropy_weight`
argument).

The setting we use in the paper, &lambda; = 30, is a very strong
weight, and is not stable for all configurations.
When the entropy weight is too large, the repulsive force from
entropy regularization may prevent InfoNCE from aligning matches.

As an example, when training SSCD using Torchvision in this
codebase, we discovered that our &lambda; = 30 results relied
on Classy Vision's default ResNet initialization, equivalent to
TorchVision ResNet's `zero_init_residual=True` option, which puts all
the energy into the residual connections at initialization.

We recommend using a lower initial weight (eg. &lambda; = 10) for new
experiments.
