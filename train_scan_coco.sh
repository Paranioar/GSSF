#----------------------- T2I baseline ----------------------------
#GPU_DEVICES='0'
#DATASET_NAME='coco'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_scan_baseline_t2i_dp
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=20 --lr_update=10 --margin 0.2 --learning_rate=.0005 \
#  --attn_name scan --attn_type t2i --attn_smooth 9.0

#----------------------- T2I vector ----------------------------
#GPU_DEVICES='1'
#DATASET_NAME='coco'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_scan_vector_t2i_dp
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=20 --lr_update=10 --margin 0.2 --learning_rate=.0005 \
#  --attn_name scan --attn_type t2i --attn_smooth 9.0 --use_sim_metric

#----------------------- T2I matrix ----------------------------
#GPU_DEVICES='2'
#DATASET_NAME='coco'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_scan_matrix_t2i_dp_sim256
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=20 --lr_update=10 --margin 0.2 --learning_rate=.0005 \
#  --attn_name scan --attn_type t2i --attn_smooth 9.0 --use_sim_matrix --sim_dim 256

#----------------------- I2T baseline ----------------------------
#GPU_DEVICES='0'
#DATASET_NAME='coco'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_scan_baseline_i2t_dp
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=20 --lr_update=10 --margin 0.2 --learning_rate=.0005 \
#  --attn_name scan --attn_type i2t --attn_smooth 4.0

#----------------------- I2T vector ----------------------------
#GPU_DEVICES='1'
#DATASET_NAME='coco'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_scan_vector_i2t_dp
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=20 --lr_update=10 --margin 0.2 --learning_rate=.0005 \
#  --attn_name scan --attn_type i2t --attn_smooth 4.0 --use_sim_metric

#----------------------- I2T matrix ----------------------------
#GPU_DEVICES='2'
#DATASET_NAME='coco'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_scan_matrix_i2t_dp_sim256
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=20 --lr_update=10 --margin 0.2 --learning_rate=.0005 \
#  --attn_name scan --attn_type i2t --attn_smooth 4.0 --use_sim_matrix --sim_dim 256