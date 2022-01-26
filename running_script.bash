#!/bin/bash

python run.py --a ViolaJones --s 256 --f 200 & python ./measure_stats.py --a ViolaJones
echo ViolaJones done
sleep 90

python run.py --a SSDResNet --s 256 --f 200 & python ./measure_stats.py --a SSDResNet
echo SSDResNet done
sleep 90

python run.py --a DLibHogSVM --s 256 --f 200 & python ./measure_stats.py --a DLibHogSVM
echo DLibHogSVM done
sleep 90

python run.py --a DLibMMOD --s 256 --f 200 & python ./measure_stats.py --a DLibMMOD
echo DLibMMOD done
sleep 90

python run.py --a MTCNN --s 256 --f 200 & python ./measure_stats.py --a MTCNN
echo MTCNN done
sleep 90

python run.py --a SSDMobileNet --s 256 --f 200 & python ./measure_stats.py --a SSDMobileNet
echo SSDMobileNet done
sleep 90

python run.py --a TinyYolo --s 256 --f 200 & python ./measure_stats.py --a TinyYolo
echo TinyYolo done
sleep 90

python run.py --a YoloV2 --s 256 --f 200 & python ./measure_stats.py --a YoloV2
echo YoloV2 done
sleep 90

python run.py --a YoloV3 --s 256 --f 200 & python ./measure_stats.py --a YoloV3
echo YoloV3 done

echo Done
