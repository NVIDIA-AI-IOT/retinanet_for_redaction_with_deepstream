# Training

These instructions show how to use the NVIDIA PyTorch implementation of RetinaNet to train a face detection model,
which we can then use of redact faces in video streams.

## Training setup

```bash
DATA_DIR=/<path to your data dir>
WORKING_DIR=/<path to this directory>
docker run -it --gpus all --rm --ipc=host -v$DATA_DIR:/data -v$WORKING_DIR:/src -w/src nvcr.io/nvidian/pytorch:19.06-py3
```

Install `retinanet` from the NVIDIA implementation repository.

```bash
pip install --no-cache-dir git+https://github.com/nvidia/retinanet-examples
```

## Train

We assume that your data has been pre-processed, as described in the [data README](DATA_README.md). An example training 
command is shown below.

```bash
retinanet train redaction.pth --backbone ResNet18FPN --fine-tune retinanet_rn18fpn.pth  --classes 1 \ 
        --lr 0.0001  --batch 80 \ 
        --images /data/open_images/train_faces  --annotations /data/open_images/train_faces.json  \
        --val-images /data/open_images/validation --val-annotations /data/open_images/val_faces.json \
        --val-iters 5000  --max-size 880 --iters 50000 --milestones 30000 40000
```

