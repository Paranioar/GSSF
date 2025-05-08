#GPU_DEVICES='0'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_bfan_baseline_equal_dp
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=15 --margin 0.2 --learning_rate=.0002 \
#  --attn_name bfan --attn_type equal --attn_smooth 20.0
#
#
#GPU_DEVICES='0'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_bfan_vector_equal_dp
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=15 --margin 0.2 --learning_rate=.0002 \
#  --attn_name bfan --attn_type equal --attn_smooth 20.0 --use_sim_metric
#
#
#GPU_DEVICES='4'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_bfan_matrix_equal_dp_sim256
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=15 --margin 0.2 --learning_rate=.0002 \
#  --attn_name bfan --attn_type equal --attn_smooth 20.0 --use_sim_matrix --sim_dim 256


#GPU_DEVICES='1'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_bfan_baseline_prob_dp
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=15 --margin 0.2 --learning_rate=.0002 \
#  --attn_name bfan --attn_type prob --attn_smooth 20.0
#
#
#GPU_DEVICES='1'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_bfan_vector_prob_dp
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=15 --margin 0.2 --learning_rate=.0002 \
#  --attn_name bfan --attn_type prob --attn_smooth 20.0 --use_sim_metric
#
#
#GPU_DEVICES='1'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_bfan_matrix_prob_dp_sim256
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=15 --margin 0.2 --learning_rate=.0002 \
#  --attn_name bfan --attn_type prob --attn_smooth 20.0 --use_sim_matrix --sim_dim 256