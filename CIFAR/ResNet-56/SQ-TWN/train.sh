BASE_PATH=$(pwd)
TRAIN_PATH=$(dirname $0)

cd $TRAIN_PATH
mkdir model

TOOLS=$BASE_PATH/caffe/build/tools

$TOOLS/caffe train \
    --solver=solver_s1.prototxt -gpu 0

$TOOLS/caffe train \
    --solver=solver_s2.prototxt -gpu 0 \
	--weights=model/cifar_resnet56_SQTWN_s1_iter_64000.caffemodel

$TOOLS/caffe train \
    --solver=solver_s3.prototxt -gpu 0 \
	--weights=model/cifar_resnet56_SQTWN_s2_iter_64000.caffemodel

$TOOLS/caffe train \
    --solver=solver_s4.prototxt -gpu 0 \
	--weights=model/cifar_resnet56_SQTWN_s3_iter_64000.caffemodel
