# Data preparation

These instructions show how to convert [Open Images v5](https://storage.googleapis.com/openimages/web/index.html) 
annotations into a COCO format dataset that we can use for face detection.

## Enter container

We will work in a PyTorch container, which we download from the [NVIDIA GPU Cloud](https://ngc.nvidia.com). 
If you don't have one, sign up for a free account and create an API KEY.

Login to `nvcr.io`
```bash
docker login nvcr.io
```

Now we can enter the container

```bash
DATA_DIR=/<path to your data dir>
WORKING_DIR=/<path to this directory>
docker run -it --gpus all --rm --ipc=host -v$DATA_DIR:/data -v$WORKING_DIR:/src -w/src nvcr.io/nvidian/pytorch:19.09-py3
```

## Download Open Images

Download the dataset by running the download script from the data directory.
```bash
cd /data/open_images
bash /src/open_images/download_open_images.sh
bash /src/open_images/unzip_open_images.sh
```

Your `/data` directory should look like this:

```
>> du -sh *
10G     challenge2018
4.0K    challenge-2018-attributes-description.csv
4.0K    challenge-2018-relationships-description.csv
12K     challenge-2018-relationship-triplets.csv
12K     class-descriptions-boxable.csv
37G     test
74M     test-annotations-bbox.csv
31M     test-annotations-human-imagelabels-boxable.csv
15M     test-images.csv
44M     test-images-with-rotation.csv
60G     train_00
60G     train_01
60G     train_02
60G     train_03
60G     train_04
59G     train_05
59G     train_06
60G     train_07
43G     train_08
1.2G    train-annotations-bbox.csv
360M    train-annotations-human-imagelabels-boxable.csv
207M    train-images-boxable.csv
609M    train-images-boxable-with-rotation.csv
13G     validation
24M     validation-annotations-bbox.csv
11M     validation-annotations-human-imagelabels-boxable.csv
5.2M    validation-images.csv
15M     validation-images-with-rotation.csv
570G    zips
```

## Parse validation data
We want to produce a `.json` file that contains all the images from some classes, and a subset of images from the other classes.

Working in the `/src` directory, we start by defining the Open Images validation images and annotation files, and the location of our output data.
```python
images_dir = '/data/open_images/validation'
annotation_csv = '/data/open_images/validation-annotations-bbox.csv'
category_csv = '/data/open_images/class-descriptions-boxable.csv'
output_json = '/data/open_images/val_faces.json'

# Now we read the Open Images categories and parse our data.

import open_images.open_image_to_json as oij
from data_tools.coco_tools import write_json
catmid2name = oij.read_catMIDtoname(category_csv)
oidata = oij.parse_open_images(annotation_csv) # This is a representation of our dataset.

# We only want images that contain the 'Human Face' class, so we run a function that removes all other images.

set1 = oij.reduce_data(oidata, catmid2name, keep_classes=['Human face'])

# Finally we convert this data to COCO format, using this as an opportunity to exclude any annotations 
# that are smaller than 2 x 2 when the input images are resized to maxdim 640, and save to a file.

cocodata = oij.openimages2coco(set1, catmid2name, images_dir, 
                               desc="Open Image validation data, set 1.", 
                               output_class_ids={'Human face': 1}, 
                               max_size=880, min_ann_size=(1,1), 
                               min_ratio=2.0)
oij.write_json_data(cocodata, output_json)

```

## Parse training data
Following the same process, we can produce a training dataset. 

```python
import open_images.open_image_to_json as oij

# Definine paths
images_dir = ['/data/open_images/train_0%i'%oo for oo in range(9)] # There are nine image directories.
annotation_csv = '/data/open_images/train-annotations-bbox.csv'
category_csv = '/data/open_images/class-descriptions-boxable.csv'
output_json = '/data/open_images/train_faces.json'

# Read the category names
catmid2name = oij.read_catMIDtoname(category_csv)
# Parse the annotations
oidata = oij.parse_open_images(annotation_csv)

# Keep only human faces
trainset1 = oij.reduce_data(oidata, catmid2name, keep_classes=['Human face'])
cocodata = oij.openimages2coco(trainset1, catmid2name, images_dir, desc="Open Image train data, set 1.", 
                               output_class_ids={'Human face': 1}, 
                               max_size=880, min_ann_size=(1,1), 
                               min_ratio=2.0)
oij.write_json_data(cocodata, output_json)

```

## Copy images in our dataset

Copy images that are in our dataset, from the Open Images directories to a new directory.


```python
import open_images.open_image_to_json as oij
oij.copy_images('/data/open_images/val_faces.json', 
                 '/data/open_images/validation', '/data/open_images/val_faces')
images_dir = ['/data/open_images/train_0%i'%oo for oo in range(9)] # There are nine image directories.          
oij.copy_images('/data/open_images/train_faces.json', images_dir, 
                '/data/open_images/train_faces')
```

## Plot ground truth

As a quick sanity check, let's plot some of our training set.

```python
from data_tools.plot_images import draw_boxes
image_dir = '/data/open_images/train_faces'
anns = '/data/open_images/processed_train/train_faces.json'
output_dir = '/data/open_images/gt_plot_train_faces'
draw_boxes(image_dir, output_dir, anns)
```