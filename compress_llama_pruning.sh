#! /bin/bash
GPUS_PER_NODE=2

NNODES=1
MASTER_ADDR="localhost"
MASTER_PORT=23344
MODEL_STEPS="0"

python /root/zhaoyq/compress_llama/update_dataset_config.py /root/zhaoyq/compress_llama/datasets/_datasets.json

OPTS=""
OPTS+=" --model-config /root/zhaoyq/models/llama-7b-bmt"
OPTS+=" --dataset=/root/zhaoyq/compress_llama/datasets/_datasets.json"
# OPTS+=" --dataset=datasets/_datasets.json"
OPTS+=" --batch-size 16"
OPTS+=" --train-iters 200000"
OPTS+=" --save-iters 500"
OPTS+=" --save-name llama-7b-prune"
OPTS+=" --save-opt-name cpm_live_checkpoint"
OPTS+=" --max-length 1024"
OPTS+=" --save /root/zhaoyq/models/cook/checkpoint"
OPTS+=" --lr 0.000005"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-iters 2000"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 0.1"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 1048576"
OPTS+=" --start-step 0"
OPTS+=" --log-dir /root/zhaoyq/models/cook/checkpoint/logs"
OPTS+=" --tensorboard /root/zhaoyq/tensorboard_log/llama_cook"
OPTS+=" --load /root/zhaoyq/models/llama-7b-bmt/pytorch_model.pt"

OPTS+=" --cook-config /root/zhaoyq/compress_llama/config/prune.json"
OPTS+=" --cook-save /root/zhaoyq/models/cook/checkpoint"
OPTS+=" --cook-save-name llama-7b-prune"
OPTS+=" --cook-mask-save /root/zhaoyq/models/cook/masks"
OPTS+=" --cook-save-mask-iters 0" # nop
OPTS+=" --cook-save-iters 500"
OPTS+=" --cook-save-mode prune"
OPTS+=" --log-file /root/zhaoyq/logs/llama-7b-prune.log"
OPTS+=" --load-teacher nop"
OPTS+=" --teacher-config nop"

CMD="torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} /root/zhaoyq/compress_llama/compress_llama.py ${OPTS}"

echo ${CMD}
$CMD | tee /root/zhaoyq/logs/llama-7b-print-prune.log
