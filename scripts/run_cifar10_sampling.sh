set -x

EXP_PREFIX=$1
SAMPLING_STRATEGY=$2
NET=$3
BATCH_SIZE=$4
LR=$5
DECAY=$6
MAX_NUM_BACKPROPS=$7
SAMPLING_MIN=$8

EXP_NAME=$EXP_PREFIX"_"$SAMPLING_STRATEGY

mkdir "/proj/BigLearning/ahjiang/output/cifar10/"
OUTPUT_DIR="/proj/BigLearning/ahjiang/output/cifar10/"$EXP_NAME
PICKLE_DIR=$OUTPUT_DIR/pickles
mkdir $OUTPUT_DIR
mkdir $PICKLE_DIR

NUM_TRIALS=3
for i in `seq 1 $NUM_TRIALS`
do
  OUTPUT_FILE="sampling_cifar10_"$NET"_"$SAMPLING_MIN"_"$BATCH_SIZE"_"$LR"_"$DECAY"_trial"$i"_v3"
  PICKLE_PREFIX="sampling_cifar10_"$NET"_"$SAMPLING_MIN"_"$BATCH_SIZE"_"$LR"_"$DECAY"_trial"$i

  echo $OUTPUT_DIR/$OUTPUT_FILE

  python main.py \
    --sb-strategy=sampling \
    --net=$NET \
    --batch-size=$BATCH_SIZE \
    --decay=$DECAY \
    --max-num-backprops=$MAX_NUM_BACKPROPS \
    --pickle-dir=$PICKLE_DIR \
    --pickle-prefix=$PICKLE_PREFIX \
    --sampling-min=$SAMPLING_MIN \
    --sampling-strategy=$SAMPLING_STRATEGY \
    --lr $LR &> $OUTPUT_DIR/$OUTPUT_FILE
done
