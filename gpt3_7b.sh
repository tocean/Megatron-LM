#! /bin/bash
# sh gpt_3b.sh
# PP=1 TP=4 BS=8
# sh debug.sh <nnodes> <node rank> <gpus per node> <mini batch size> <PP> <TP> <ADDR>

if [ "$1" != "" ]; then
  echo "total nodes $1"
  NNODES=$1
else
  echo "this is master node and node rank = 0"
  NNODES=1
fi

if [ "$2" != "" ]; then
  echo "this is node $2"
  NODE_RANK=$2
else
  echo "this is master node and node rank = 0"
  NODE_RANK=0
fi

if [ "$3" != "" ]; then
  echo "gpus per node $3"
  GPUS_PER_NODE=$3
else
  echo "this is master node and node rank = 0"
  GPUS_PER_NODE=8
fi

if [ "$4" != "" ]; then
  echo "Using mini batch size $4"
  BS=$4
else
  echo "Using mini batch size 1"
  BS=8
fi

if [ "$5" != "" ]; then
  echo "Using pipline parallel $5"
  PP=$5
else
  echo "Using pipline parallel 1"
  PP=1
fi

if [ "$6" != "" ]; then
  echo "Using tensor parallel $6"
  TP=$6
else
  echo "Using tensor parallel 1"
  TP=4
fi

if [ "$7" != "" ]; then
  echo "Using tensor parallel $7"
  MASTER_ADDR="$7"
else
  echo "Using tensor parallel 1"
  MASTER_ADDR=127.0.0.1
fi

# wandb login [WANDB-KEY]

MASTER_PORT=6001
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
MBS=$(($WORLD_SIZE * $BS / $PP / $TP))

echo "global min batch size = $MBS"

# DATA_PATH=/data/mnt/gpt_data
DATA_PATH=/hostroot/mnt/nvme1n1/kanwu/gpt-data/
DATA_PATH_1=$DATA_PATH/stories_meg_gpt_document
DATA_PATH_2=$DATA_PATH/bookcorpus_meg_gpt_document
DATA_PATH_3=$DATA_PATH/ccnews_meg_gpt_document
DATA_PATH_4=$DATA_PATH/Pile_CC_meg_gpt_document
DATA_PATH_5=$DATA_PATH/OpenWebText2_meg_gpt_document
DATA_PATH_6=$DATA_PATH/USPTO_meg_gpt_document
DATA_PATH_7=$DATA_PATH/GutenbergProject_meg_gpt_document
DATA_PATH_8=$DATA_PATH/Wikipedia_en_meg_gpt_document
DATA_PATH_9=$DATA_PATH/DM_Math_meg_gpt_document
DATA_PATH_10=$DATA_PATH/HackerNews_meg_gpt_document
#DATASET="0.05 ${DATA_PATH_1} 0.01 ${DATA_PATH_2} 0.13 ${DATA_PATH_3} \
#            0.51 ${DATA_PATH_4} 0.04 ${DATA_PATH_5} 0.07 ${DATA_PATH_6}\
#            0.05 ${DATA_PATH_7} 0.08 ${DATA_PATH_8} 0.05 ${DATA_PATH_9} 0.01 ${DATA_PATH_10}"
DATASET="1.0 ${DATA_PATH_3}"

RUN=gpt3m-opt-data-fp16-kan-debug-lab2-8GPUs-TP$TP-PP$PP
#CHECKPOINT_PATH=/data/mnt/azsussc/wukan/tiny_vit/gpt_checkpoint/$RUN
CHECKPOINT_PATH=./debug_checkpoints/$RUN
mkdir -p $CHECKPOINT_PATH

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"


export CUDA_DEVICE_MAX_CONNECTIONS=1
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
  pretrain_gpt.py \
  --tensor-model-parallel-size $TP \
  --pipeline-model-parallel-size $PP \
  --num-layers 32 \
  --hidden-size 4096 \
  --num-attention-heads 32 \
  --seq-length 2048 \
  --max-position-embeddings 2048 \
  --micro-batch-size $BS \
  --global-batch-size 256 \
  --rampup-batch-size $MBS $MBS 1953125 \
  --train-samples 146484375 \
  --lr-decay-samples 126953125 \
  --lr-warmup-samples 183105 \
  --lr 3.0e-4 \
  --min-lr 3.0e-5 \
  --lr-decay-style cosine \
  --clip-grad 1.0 \
  --weight-decay 0.1 \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --adam-eps 1e-8 \
  --init-method-std 0.02 \
  --fp16 \
  --log-interval 16 \
  --eval-iters 35 \
  --eval-interval 5120 \
  --save-interval 20480 \
  --save $CHECKPOINT_PATH \
  --data-path $DATASET \
  --vocab-file gpt2-vocab.json \
  --merge-file gpt2-merges.txt \
  --split 949,50,1 \
  --distributed-backend nccl \
  --use-flash-attn \
  --no-query-key-layer-scaling \
  --use-distributed-optimizer \
  --tensorboard-log-interval 128 \
  --fp8-hybrid \
  --transformer-impl transformer_engine

#  --recompute-granularity full \
#  --recompute-method block \
#  --recompute-num-layers 10000000 \

#  --no-async-tensor-model-parallel-allreduce \
#  --wandb-run $RUN \
#  --wandb-entity gpt-style \
#  --use_fp8_linear \
#  --tensorboard-dir ./tb \

#
#> ${LOG_PATH}/run_${Name}_${NODE_RANK}.log 2>&1 &   \

# --data-impl mmap \
