We are using [Qualcomm Neural Processing SDK][1] to quantize our model.

### Instructions to Quantize

1. Start a terminal session and run the docker environment in which you have set up the SDK:

```
sudo docker start snpe_qnn_container
sudo docker exec -it snpe_qnn_container /bin/bash
```

2. Update the environment variables to point to the correct file paths on your setup:

```
export QNN_SDK_ROOT=/local/
export ANDROID_NDK_ROOT=/usr/android-ndk-r26c
export PATH="${ANDROID_NDK_ROOT}:${PATH}"
export TENSORFLOW_HOME=/local/2.37.0.250724/bin/venv/lib/python3.10/site-packages/tensorflow
source /local/2.37.0.250724/bin/envsetup.sh
```

3. Activate the python virtual environment:

```
source /local/2.37.0.250724/bin/venv/bin/activate
```

4. Navigate to your work directory:

```
cd local/2.37.0.250724/examples/Models/work-dir/
```

5. Make sure that correct files are present in the directory:

```
(venv) root@hostname:/local/2.37.0.250724/examples/Models/work-dir# ls 
convert_csv_to_raw.py  execute_models.sh  frozen_model  gestures_test.csv  gestures_train.csv
```

6. Set your host machine architecture:

```
export HOST_MACHINE_ARCH="x86_64-linux-clang"
```

7. Build the .dlc file:

```
qairt-converter \
  --input_network frozen_model/frozen_graph.pb \
  --source_model_input_shape 'x' 1,126 \
  --out_tensor_node 'sequential_1/dense_2_1/Softmax' \
  --output_path mymodel_float32.dlc
```

8. Build raw binaries from the .csv data:

```
(venv) root@hostname:/local/2.37.0.250724/examples/Models/work-dir# python3 convert_csv_to_raw.py 
Enter the path to your .csv data file: gestures_test.csv
Enter the name for the output directory: gestures_test_raw
Loading data from: gestures_test.csv
Saving .raw files to: gestures_test_raw/
✅ Successfully created .raw files.
✅ input_list.txt with correct relative paths saved to: gestures_test_raw/input_list.txt
```

```
(venv) root@hostname:/local/2.37.0.250724/examples/Models/work-dir# python3 convert_csv_to_raw.py 
Enter the path to your .csv data file: gestures_train.csv
Enter the name for the output directory: gestures_train_raw
Loading data from: gestures_train.csv
Saving .raw files to: gestures_train_raw/
✅ Successfully created .raw files.
✅ input_list.txt with correct relative paths saved to: gestures_train_raw/input_list.txt
```

9. Quantize the model:

```
qairt-quantizer \
  --input_dlc mymodel_float32.dlc \
  --input_list gestures_train_raw/input_list.txt \
  --output_dlc mymodel_quantized_int8.dlc
```

```
qairt-quantizer \
  --input_dlc mymodel_float32.dlc \
  --input_list gestures_train_raw/input_list.txt \
  --weights_bitwidth 8 \
  --act_bitwidth 16 \
  --output_dlc mymodel_quantized_w8a16.dlc
```

### Instructions to Run on QIDK

10. On a different terminal session, connect to qidk:

```
adb shell
```

11. [Target Device] Set the environment variable for the work directory:

```
export DESTINATION="/data/local/tmp/work_dir"
```

12. [Target Device] Set additional environment variables:

```
export PATH=$PATH:${DESTINATION}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${DESTINATION}
export ADSP_LIBRARY_PATH="${DESTINATION};/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"
```

13. [Host Device] Set the environment variable for the work directory on Target Device: 

```
export DESTINATION="/data/local/tmp/work_dir"
```

14. [Host Device] Set the Target Device architecture:

```
export TARGET_DEVICE_ARCH="aarch64-android"
```

15. [Host Device] Transfer all the model related files to Target Device:

```
adb push "${SNPE_ROOT}/examples/Models/work-dir/mymodel_float32.dlc"  "${DESTINATION}"
adb push "${SNPE_ROOT}/examples/Models/work-dir/mymodel_quantized_int8.dlc"  "${DESTINATION}"
adb push "${SNPE_ROOT}/examples/Models/work-dir/mymodel_quantized_w8a16.dlc"  "${DESTINATION}"
adb push "${SNPE_ROOT}/examples/Models/work-dir/gestures_test_raw"  "${DESTINATION}"
adb push "${SNPE_ROOT}/examples/Models/work-dir/execute_models.sh"  "${DESTINATION}"
```

16. [Host Device] Transfer the runtimes to Target Device:

```
adb push "$SNPE_ROOT/lib/${TARGET_DEVICE_ARCH}/libSNPE.so"  "${DESTINATION}"
adb push "$SNPE_ROOT/bin/$TARGET_DEVICE_ARCH/snpe-net-run"  "${DESTINATION}"
```

17. [Host Device] Set the Hexagon architecture value:

```
export HEXAGON_VERSION="75"
export HEXAGON_ARCH="hexagon-v${HEXAGON_VERSION}"
```

18. [Host Device] Transfer the DSP related runtimes to Target Device:

```
adb push "$SNPE_ROOT/lib/${HEXAGON_ARCH}/unsigned/libSnpeHtpV${HEXAGON_VERSION}Skel.so"  "${DESTINATION}"
adb push "$SNPE_ROOT/lib/${TARGET_DEVICE_ARCH}/libSnpeHtpV${HEXAGON_VERSION}Stub.so"  "${DESTINATION}"
adb push "$SNPE_ROOT/lib/${TARGET_DEVICE_ARCH}/libSnpeHtpPrepare.so"  "${DESTINATION}"
```

19. [Target Device] Execute the models on different hardware units:

```
chmod +x execute_models.sh
./execute_models.sh
```




[1]:https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/overview.html?product=1601111740010412