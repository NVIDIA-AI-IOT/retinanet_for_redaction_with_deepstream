# Face Redaction with DeepStream

This sample shows how to train and deploy a deep learning model for the real time redaction of faces from video streams using the NVIDIA DeepStream SDK.

## Blog posts
There are two blog posts that accompany this repo:
- [Building a Real-time Redaction App Using NVIDIA DeepStream, Part 1: Training](https://devblogs.nvidia.com/real-time-redaction-app-nvidia-deepstream-part-1-training/)
- [Building a Real-time Redaction App Using NVIDIA DeepStream, Part 2: Deployment](https://devblogs.nvidia.com/real-time-redaction-app-nvidia-deepstream-part-2-deployment/)

## Getting Started Guide

### Data preparation

The [data README](DATA_README.md) explains how we used python to convert 
[Open Images v5](https://storage.googleapis.com/openimages/web/index.html) annotations into COCO format annotations. 

### Training using RetinaNet

The [training README](TRAINING_README.md) shows how to train, evaluate and export a model using the [NVIDIA PyTorch implementation of RetinaNet](https://github.com/NVIDIA/retinanet-examples).

### Redaction app setup

#### DeepStream
Before we start with our redaction app, please make sure that you're able to successfully run the DeepStream sample apps. 
From the DeepStream `samples` folder, you can call any of the config files (`source*.txt`) in `configs/deepstream-app/`.
```
deepstream-app -c configs/deepstream-app/source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_int8.txt 
```
Running the example above will show four streams of the same video, with vehicles and pedestrians being detected. If the performance of sample apps such as is lagging, try running *sudo /usr/bin/jetson_clocks* to set max performance.

#### Making the redaction app

Now we can make the redaction app.

* Copy the contents of this folder to `<path to deepstream>/deepstream_sdk_v4.0_jetson/sources/apps/`.
* Install the prerequisites.

```bash
    apt install libssl1.0.0 libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \ 
        gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstrtspserver-1.0-0 libjansson4=2.11-1 \ 
        librdkafka1=0.11.3-1build1
```

* Run `make`
  * If `make` gives any error, check that 1) in the Makefile we are pointing to correct and valid paths, and 2) the appropriate packages have been installed, including libgstreamer1.0-dev, libgstreamer1.0, and libgstreamer-plugins-base1.0-dev.


### Using the redaction app

```bash
./deepstream-redaction-app -c configs/test_source*_<precision>.txt
```

However, before we can run our app with a config file, we'll want to modify a few parameters in the config files. We will introduce the parameters in the following section.


## DeepStream config files

### 1. Main pipeline config file

Examples in our config/ folder include: test_source1_fp16, test_source4_fp16, test_source8_fp16, test_source1_int8, test_source4_int8, test_source8_int8. For official DeepStream examples, please see config files in folder `<path to deepstream>/deepstream_sdk_v4.0_jetson/samples/configs/deepstream-app/`.

The parameter `enable` turns each of the pipeline elements on and off.

#### source	

Change `[source*]` for the type of source you want to stream from. Type 1 is from a live camera connected to Jetson. Type 2 URI can be a local mp4 file or a rtsp source. Type 3 MultiURI pertains to the case where we want to have more than one source streaming.

If we want to stream a local mp4 as the source, we can change the uri parameter in `[source*]` to point to the path of that mp4 file. For example, if the mp4 file is located in /home/this_user/videos/, we would want to write uri=file:///home/this_user/videos/your-file.mp4 

#### sink

Change `[sink*]` for the type of output sink you want the stream to go to. 

#### multiple streams	

For multiple streams we need to modify and add `[source*]` as necessary. We will also need to change: the rows and columns number in `[tiled-display]`, the batch-size in `[streammux]` to equal the number of sources, and the batch-size in `[primary-gie]` as well. If we see a performance drop as the number of streams increases, one adjustment we can make to meet real-time criteria is to modify the interval parameter in `[primary-gie]`:
if we set `interval = 1`, that means we're inferring every other frame instead of every single frame (when `interval = 0`). When we set interval > 0, we should turn on the `[tracker]` to track objects of interest.

#### model engine file	

The model-engine-file parameter in [primary-gie] should point to the TRT model engine file to be deployed. Find the path to your engine.plan file that you generated on this Jetson device (this file was generated using the `./export` command in the retinanet repo's `cppapi` folder). We can copy the file over to the `odtk_models` folder for clarity.

#### config file

The config-file parameter in [primary-gie] should point to the model config file.
	
#### loop

If you are streaming mp4 files or rtsp sources of a finite time period and would like to run an infinite loop on the source, you could change the following "file-loop" parameter to 1 in the config file.
```
[tests]
file-loop=0
```
#### osd
    
In this app, our goal is to redact faces, therefore our on screen display (osd) doesn't need to display anything other than a box. If you want to display texts or more, please see example config files at `<path to deepstream>/deepstream_sdk_v4.0_jetson/samples/configs/deepstream-app/`. To get started, you'd want to modify the parameters in [osd] and [primary-gie]. 

### 2. Model config file

Examples in our config/ folder include: odtk_model_config_int8.txt and odtk_model_config_fp16.txt.

The parameter network-mode in [property] should correspond to the precision of your TensorRT model engine file. 

We need to make sure that custom-lib-path in [property] is pointing to the correct file path for the custom library `libnvdsparsebbox_retinanet.so` we just built. 

The parameter threshold in [class-attrs-all] will decide the threshold above which detections are outputted. You can modify the value of threshold to have more detections or less.


For the provided config file examples in folder configs/, we will want to modify at least the following to get the app running:

1. In odtk_model_config_*.txt, the custom-lib-path parameter in [property], which should point to the `libnvdsparsebbox_retinanet.so` file that we built.

2. In test_source*.txt, the model-engine-file parameter in [primary-gie], which should point to an engine plan file.
	
3. In test_source*.txt, the uri parameter in [source*], which points to the path to the mp4 file if we are inferring on a mp4 file.

	


### Possible runtime errors

 
```
pp:511:gst_nvinfer_logger:<primary_gie_classifier> NvDsInferContext[UID 1]:generateTRTModel(): No model files specified
```
	

This is caused by not having a valid model engine file in the config file. There are some possibilities: it's either that we do not have the file with the specified name in the path, or, we have a model engine in this path, but the batch size that the model engine has is smaller than then number of sources/streams we are trying to process. For example, if we exported the onnx file to a engine.plan file using an executable made with batch size = 4 specified in `export.cpp`, then our engine cannot process more than 4 streams at a time.



