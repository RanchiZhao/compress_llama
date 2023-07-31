#! /bin/bash
GPUS_PER_NODE=8

NNODES=1
MASTER_ADDR="localhost"
MASTER_PORT=23335
MODEL_STEPS="0"

# BASE="/data"
BASE="/local/apps"
# TMP="/data/compress_llama"
TMP="/data"
METHOD="relu"

mkdir -p ${TMP}/checkpoints/logs/${METHOD}
mkdir -p ${TMP}/checkpoints/${METHOD}
mkdir -p ${TMP}/checkpoints/mask
pip install scikit-learn
pip install bmtrain==0.2.2
pip install sentencepiece
pip install transformers==4.28.0
pip install model-center==1.0.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install cpm-live==0.1.0
pip install cpm-kernels
pip install bmcook==0.1.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
ls
pwd

echo ${PLATFORM_CONFIG_PATH}
mkdir -p /data/config
python update_dataset_config.py datasets/_datasets.json
echo "here"
cat /data/config/_datasets.json

pip list
which pip
which python
which torchrun

OPTS=""
OPTS+=" --model-config /mnt/data/user/tc_agi/user/zhaoweilin/llama-7b"
OPTS+=" --dataset=/data/config/_datasets.json"
# OPTS+=" --dataset=datasets/_datasets.json"
OPTS+=" --batch-size 16"
OPTS+=" --train-iters 200000"
OPTS+=" --save-iters 500"
OPTS+=" --save-name llama-7b-${METHOD}"
OPTS+=" --save-opt-name cpm_live_checkpoint"
OPTS+=" --max-length 1024"
OPTS+=" --save ${TMP}/checkpoints/${METHOD}"
OPTS+=" --lr 0.000005"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-iters 2000"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 0.1"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 1048576"
OPTS+=" --start-step ${MODEL_STEPS}"
OPTS+=" --log-dir ${TMP}/checkpoints/logs/${METHOD}"
OPTS+=" --tensorboard ${TMP}/logs/tensorboard/${METHOD}"
OPTS+=" --load /mnt/data/user/tc_agi/user/zhaoweilin/llama-7b/pytorch_model.pt"

OPTS+=" --cook-config ${BASE}/compress_llama/config/${METHOD}.json"
OPTS+=" --cook-save ${TMP}/checkpoints/${METHOD}"
OPTS+=" --cook-save-name llama-7b-${METHOD}"
OPTS+=" --cook-mask-save ${TMP}/checkpoints/masks/mask.bin"
OPTS+=" --cook-save-mask-iters 0" # nop
OPTS+=" --cook-save-iters 500"
OPTS+=" --cook-save-mode ${METHOD}"
OPTS+=" --log-file ${TMP}/checkpoints/logs/llama-7b-${METHOD}.log"
OPTS+=" --load-teacher nop"
OPTS+=" --teacher-config nop"

CMD="torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} compress_llama.py ${OPTS}"

echo ${CMD}
$CMD | tee ${TMP}/checkpoints/logs/print_${METHOD}.log
