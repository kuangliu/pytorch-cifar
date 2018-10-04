set -x

EXP_NAME=$1
NET=$2
POOL_SIZE=$3
LR=$4
DECAY=$5
MAX_NUM_BACKPROPS=$6
SAMPLING_MIN=$7

mkdir "/proj/BigLearning/ahjiang/output/cifar10/"
OUTPUT_DIR="/proj/BigLearning/ahjiang/output/cifar10/"$EXP_NAME
PICKLE_DIR=$OUTPUT_DIR/pickles
mkdir $OUTPUT_DIR
mkdir $PICKLE_DIR

NUM_TRIALS=3
for i in `seq 1 $NUM_TRIALS`
do
  OUTPUT_FILE="sampling_cifar10_"$NET"_"$SAMPLING_MIN"_"$POOL_SIZE"_"$LR"_"$DECAY"_trial"$i"_v2"
  PICKLE_PREFIX="sampling_cifar10_"$NET"_"$SAMPLING_MIN"_"$POOL_SIZE"_"$LR"_"$DECAY"_trial"$i

  echo $OUTPUT_DIR/$OUTPUT_FILE

  python main.py \
    --sb-strategy=sampling \
    --batch-size=1 \
    --net=$NET \
    --pool-size=$POOL_SIZE \
    --decay=$DECAY \
    --max-num-backprops=$MAX_NUM_BACKPROPS \
    --pickle-dir=$PICKLE_DIR \
    --pickle-prefix=$PICKLE_PREFIX \
    --sampling-min=$SAMPLING_MIN\
    --lr $LR &> $OUTPUT_DIR/$OUTPUT_FILE
done
