# Export an ONNX file into TensorRT in INT8 precision

During the first export, we need to use some images to produce a calibration table. To do this we need to modify the sample 
[export.cpp file](https://github.com/NVIDIA/retinanet-examples/blob/master/extras/cppapi/export.cpp) provided by RetinanNet.

### Code changes

Replace this code bloc...
```
const vector<string> calibration_files;
string model_name = "";
string calibration_table = argc == 4 ? string(argv[3]) : "";
```
...with specific images of your choosing. These calibration images should represent the distribution of your image space.

```
vector<string> calibration_files;
calibration_files.push_back("path-to/your-image-1");
calibration_files.push_back("path-to/your-image-2");
calibration_files.push_back("path-to/your-image-3");
    ... (the list goes on)
calibration_files.push_back("path-to/your-image-n");
	
string model_name = "ResNet34FPN"; //because we are using ResNet34 backbone
string calibration_table =  "";
```

When exporting a model from ONNX to TensorRT INT8, the `batch` parameter in `export.cpp` should be smaller than the total 
number of images `n`. Once you have a calibration table, you don't have to make it again on the same device, 
even if you're exporting to an engine of a different batch size, provided that the batch size remains smaller 
than the number of calibration images `n`.

### Create the calibration table

The command for the first export should be:
```
./export model.onnx engine.plan int8calibrationtablename
```
This command will take several minutes to run. After the first export, we should have a calibration table with a name 
similar to `Int8CalibrationTable_ResNet34FPN512x864_20`. 
We now pass the calibration table name to the command line.

### Using the table for future exports.
Before the next exports, we will want to change our export.cpp code back to 
```
const vector<string> calibration_files;
string model_name = "";
string calibration_table = argc == 4 ? string(argv[3]) : "";
```

For each engine, if we intend to have a different batch size, we will also need to remember to modify the line 
```int batch = 1;```
and re-run `make` to obtain a new `export` executable.


Now we just pass in the actual generated calibration table name. Example command:

```
./export model.onnx engine.plan Int8CalibrationTable_ResNet34FPN512x864_20
```


