**Usage Bash:**

*python run.py --a ScriptName [--s 640] [--f 10] & python ./util/measure_stats.py --a ScriptName*  
**Example**: *python run.py --a ViolaJones --s 640 --f 10 & python ./measure_stats.py --a ViolaJones*

**Usage PowerShell (PowerShell 6.0+):**

*python run.py --a ScriptName [--s 640] [--f 10] &; python ./util/measure_stats.py --a ScriptName*  
**Example**: *python run.py --a ViolaJones --s 640 --f 10 &; python ./measure_stats.py --a ViolaJones*


ScriptName Options:
* ViolaJones
* SSDResNet
* DLibHogSVM
* DLibMMOD
* MTCNN
* SSDMobileNet
* SSDMobileNetLite
* RFCN
* RFCNLite
* FasterRCNN
* FasterRCNNLite
* TinyYolo
* YoloV2
* YoloV3

--s: Resolution To Run At

--f: Frames To Process Before Saving Results