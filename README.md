
- Dataset that can be used for training: https://app.roboflow.com/andy-h1py8

**Usage run.py**: run.py [-h] -a ALGORITHM [-s SIZE] [-f FRAMES] [-sv SAVE]

optional arguments:  
  -h, --help            show this help message and exit  
  -a ALGORITHM, --algorithm ALGORITHM  
                        Algorithm To Run Inference On  
  -s SIZE, --size SIZE  Image Size To Run On  
  -f FRAMES, --frames FRAMES  
                        N_Frames To Run For  
  -sv SAVE, --save SAVE  
                        Save Timing Results  

**Collect Throughput and system data - Bash:**

*python run.py --a Algorithm [--s ImageSize] [--f nFrames] [--sv True|False] & python ./util/measure_stats.py --a Algorithm*  

**Example**: *python run.py --a ViolaJones --s 640 --f 10 --sv False & python ./measure_stats.py --a ViolaJones*

**Collect Throughput and system data - PowerShell (PowerShell 6.0+):**

*python run.py --a Algorithm [--s ImageSize] [--f nFrames] [--sv True|False] &; python ./util measure_stats.py --a Algorithm*  

**Example**: *python run.py --a ViolaJones --s 640 --f 10 --sv False &; python ./measure_stats.py --a ViolaJones*

Algorithm Options:
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