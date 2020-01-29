## Config files
In this directory, we have eight config files. 


*  `test_source1_fp16.txt`, `test_source1_int8.txt` are for running the app with one input source. The current source in these files is a live camera input.


*  `test_source4_fp16.txt`, `test_source4_int8.txt` are for running the app with four input sources. The current sources in these files are 4 different local mp4 files.


*  `test_source8_fp16.txt`, `test_source8_int8.txt` are for running the app with eight input sources. The current sources in these files is a single local mp4 files streaming eight times.


*  `odtk_model_config_fp16.txt`, `odtk_model_config_int8.txt` are config files to specify parameters about the model that is represented in TRT engine plan format. Note that running the app with these 2 config files as command line input will results in DeepStream errors.

For how to config different parameters, please see section **Config files** in the [README](../README.md) in the root folder.