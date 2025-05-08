#----------------------- Cosine ----------------------------
GPU_DEVICES='0'
DATASET_NAME='coco'
DATA_PATH='./data/'${DATASET_NAME}
VOCAB_PATH='./data/vocab'
MODEL_NAME=${DATASET_NAME}_vse_cosine_dp

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
  --num_epochs=35 --lr_update=20 --margin 0.2 --learning_rate=.0005 \
  --attn_name vse --attn_type cosine

#----------------------- Metric ----------------------------
GPU_DEVICES='0'
DATASET_NAME='coco'
DATA_PATH='./data/'${DATASET_NAME}
VOCAB_PATH='./data/vocab'
MODEL_NAME=${DATASET_NAME}_vse_metric_dp

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
  --num_epochs=35 --lr_update=20 --margin 0.2 --learning_rate=.0005 \
  --attn_name vse --attn_type none --use_sim_metric --attn_dp 0.1

#----------------------- Matrix ----------------------------
GPU_DEVICES='0'
DATASET_NAME='coco'
DATA_PATH='./data/'${DATASET_NAME}
VOCAB_PATH='./data/vocab'
MODEL_NAME=${DATASET_NAME}_vse_matrix_dp_sim2

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
  --num_epochs=35 --lr_update=20 --margin 0.2 --learning_rate=.0005 \
  --attn_name vse --attn_type none --use_sim_matrix --sim_dim 2 --attn_dp 0.1


GPU='0'