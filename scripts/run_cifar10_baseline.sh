set -x

EXP_NAME=$1
NET=$2
BS=$3
POOL_SIZE=64
LR=0.1
DECAY=0.0005

mkdir "/proj/BigLearning/ahjiang/output/cifar10/"
OUTPUT_DIR="/proj/BigLearning/ahjiang/output/cifar10/"$EXP_NAME
PICKLE_DIR=$OUTPUT_DIR/pickles
mkdir $OUTPUT_DIR
mkdir $PICKLE_DIR

NUM_TRIALS=3
for i in `seq 1 $NUM_TRIALS`
do
  OUTPUT_FILE="baseline_cifar10_"$NET"_"$BS"_"$POOL_SIZE"_"$LR"_"$DECAY"_trial"$i"_v2"
  PICKLE_PREFIX="baseline_cifar10_"$NET"_"$BS"_"$POOL_SIZE"_"$LR"_"$DECAY"_trial"$i

  echo $OUTPUT_DIR/$OUTPUT_FILE

  python main.py \
    --sb-strategy=baseline \
    --batch-size=$BS \
    --net=$NET \
    --decay=$DECAY \
    --pickle-dir=$PICKLE_DIR \
    --pickle-prefix=$PICKLE_PREFIX \
    --lr $LR &> $OUTPUT_DIR/$OUTPUT_FILE
done
