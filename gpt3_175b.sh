# NODE_RANK=$NODE_RANK
# NNODES=$NNODES
# MASTER_PORT=$MASTER_PORT
BS=1 #mini batch size
PP=8  #pipline parallel
TP=8  #tensor parallel
CLIP_GRAD=1.0
# for 8xG8
GLOBAL_BATCH_SIZE=1024

GPUS_PER_NODE=8
MASTER_ADDR=$MASTER_IP
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
MBS=$(($WORLD_SIZE * $BS / $PP / $TP))
ACCUMULATE=$(($GLOBAL_BATCH_SIZE / $MBS))
LOG_INTERVAL=1

echo "global min batch size = $MBS"

DATA_PATH=/root/GPT-style/crawl-text/gpt_data
DATA_PATH=$DATA_PATH/ccnews_meg_gpt_document
DATASET="1.0 ${DATA_PATH}"
#DATASET=./examples/pretrain/datasets_config/V2_CCv3_noWudao_LLAMA_proportion.jsonl

RUN=debug-megatron-175B-ccnews-only-h100-bf16-fp8-kan-lab11-$BS-$TP-$PP
TB_PATH=./tb
CHECKPOINT_PATH=/mnt/azsussc/yux/gpt_checkpoint/$RUN
mkdir -p $CHECKPOINT_PATH

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

export CUDA_DEVICE_MAX_CONNECTIONS=1
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
  pretrain_gpt.py \
  --tensor-model-parallel-size $TP \
  --pipeline-model-parallel-size $PP \
  --distributed-backend nccl \
  --use-flash-attn \
  --no-query-key-layer-scaling \
  --seed 43 \
  --num-layers 96 \
  --hidden-size 12288 \
  --num-attention-heads 96 \
  --seq-length 2048 \
  --train-samples 146484375 \
  --lr-decay-samples 131835938 \
  --lr-warmup-samples 4096000 \
  --lr 6.0e-5 \
  --min-lr 6.0e-6 \
  --lr-decay-style cosine \
  --micro-batch-size $BS \
  --global-batch-size $GLOBAL_BATCH_SIZE \
  --clip-grad $CLIP_GRAD \
  --weight-decay 0.1 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --init-method-std 0.0057 \
  --log-interval $LOG_INTERVAL \
  --tensorboard-log-interval $LOG_INTERVAL \
  --tensorboard-dir $TB_PATH \
  --eval-iters 7 \
  --eval-interval 10 \
  --save-interval 300 \
  --save $CHECKPOINT_PATH \
  --num-workers 1 \
  --data-path $DATASET \
  --tokenizer-name-or-path cl100k_base \
  --tokenizer-type TikToken \
  --split 949,50,1 \
  --log-timers-to-tensorboard \
  --bf16 \
  --log-params-norm \
  --log-num-zeros-in-grad \
  --tensorboard-log-interval 1 \
  --use-distributed-optimizer \
  --use-rotary-position-embeddings \
  --max-position-embeddings 2048 \
  --sequence-parallel \
  --fp8-hybrid \
  --transformer-impl transformer_engine

#  --no-position-embedding \
#  --use-distributed-optimizer \
#  --timing-log-level 2 \
#  --use_fp8_linear \
#  --sequence-parallel \
#  --checkpoint-activations \
#  --recompute-activations --recompute-num-layers 24 \
#  --blob-path $DATASET \
#  --data-impl lazy \
#  --dataloader-type distributed_sequential \
