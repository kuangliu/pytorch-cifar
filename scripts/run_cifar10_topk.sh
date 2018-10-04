set -x

EXP_NAME=$1
NET=$2
TOP_K=$3
POOL_SIZE=$4
LR=$5
DECAY=$6
MAX_NUM_BACKPROPS=$7

mkdir "/proj/BigLearning/ahjiang/output/cifar10/"
OUTPUT_DIR="/proj/BigLearning/ahjiang/output/cifar10/"$EXP_NAME
PICKLE_DIR=$OUTPUT_DIR/pickles
mkdir $OUTPUT_DIR
mkdir $PICKLE_DIR


NUM_TRIALS=2
for i in `seq 1 $NUM_TRIALS`
do
  OUTPUT_FILE="topk_cifar10_"$NET"_"$TOP_K"_"$POOL_SIZE"_"$LR"_"$DECAY"_trial"$i"_v2"
  PICKLE_PREFIX="topk_cifar10_"$NET"_"$TOP_K"_"$POOL_SIZE"_"$LR"_"$DECAY"_trial"$i

  echo $OUTPUT_DIR/$OUTPUT_FILE

  python main.py \
    --batch-size=1 \
    --top-k=$TOP_K \
    --net=$NET \
    --pool-size=$POOL_SIZE \
    --decay=$DECAY \
    --max-num-backprops=$MAX_NUM_BACKPROPS \
    --pickle-dir=$PICKLE_DIR \
    --pickle-prefix=$PICKLE_PREFIX \
    --lr $LR &> $OUTPUT_DIR/$OUTPUT_FILE
done
