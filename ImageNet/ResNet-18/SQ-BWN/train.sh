BASE_PATH=$(pwd)
TRAIN_PATH=$(dirname $0)

cd $TRAIN_PATH
mkdir model

TOOLS=$BASE_PATH/caffe/build/tools

$TOOLS/caffe train \
    --solver=solver_s1.prototxt -gpu 0

$TOOLS/caffe train \
	--solver=solver_s2.prototxt -gpu 0 \
	--weights=model/resnet18_SQBWN_s1_iter_300000.caffemodel

$TOOLS/caffe train \
	--solver=solver_s3.prototxt -gpu 0 \
	--weights=model/resnet18_SQBWN_s2_iter_300000.caffemodel

$TOOLS/caffe train \
	--solver=solver_s4.prototxt -gpu 0 \
	--weights=model/resnet18_SQBWN_s3_iter_300000.caffemodel

