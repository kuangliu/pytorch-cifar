set -x

EXP_NAME=$1
NET=$2
TOP_K=$3
POOL_SIZE=$4
LR=$5

mkdir "/proj/BigLearning/ahjiang/output/cifar10/"
OUTPUT_DIR="/proj/BigLearning/ahjiang/output/cifar10/"$EXP_NAME
mkdir $OUTPUT_DIR
OUTPUT_FILE="cifar10_"$NET"_"$TOP_K"_"$POOL_SIZE"_"$LR".v1"

python main.py \
  --selective-backprop=True \
  --batch-size=1 \
  --top-k=$TOP_K \
  --net=$NET \
  --lr $LR &> $OUTPUT_DIR/$OUTPUT_FILE
