#----------------------- SGR baseline ----------------------------
#GPU_DEVICES='0'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_sgr_baseline_dp
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=40 --lr_update=30 --margin 0.2 --learning_rate=.0002 --attn_dim 256 --attn_smooth 9.0 \
#  --attn_name sgraf --attn_type sgr --attn_layer 3

#----------------------- SAF baseline ----------------------------
#GPU_DEVICES='1'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_saf_baseline_dp
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=20 --margin 0.2 --learning_rate=.0002 --attn_dim 256 --attn_smooth 9.0 \
#  --attn_name sgraf --attn_type saf

#----------------------- SGR vector ----------------------------
#GPU_DEVICES='2'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_sgr_vector_dp_qkv
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=40 --lr_update=30 --margin 0.2 --learning_rate=.0002 --attn_dim 256 --attn_smooth 9.0 \
#  --attn_name sgraf --attn_type sgr --attn_layer 3 --use_sim_metric

#----------------------- SAF vector ----------------------------
#GPU_DEVICES='3'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_saf_vector_dp_qkv
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=20 --margin 0.2 --learning_rate=.0002 --attn_dim 256 --attn_smooth 9.0 \
#  --attn_name sgraf --attn_type saf --use_sim_metric

#----------------------- SGR matrix ----------------------------
#GPU_DEVICES='0'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_sgr_matrix_dp_sim256_qkv
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=40 --lr_update=30 --margin 0.2 --learning_rate=.0002 --attn_dim 256 --attn_smooth 9.0 \
#  --attn_name sgraf --attn_type sgr --attn_layer 3 --use_sim_matrix --sim_dim 256

#----------------------- SAF matrix ----------------------------
#GPU_DEVICES='1'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_saf_matrix_dp_sim256_qkv
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=20 --margin 0.2 --learning_rate=.0002 --attn_dim 256 --attn_smooth 9.0 \
#  --attn_name sgraf --attn_type saf --use_sim_matrix --sim_dim 256