#GPU_DEVICES='0'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_naaf_baseline_dp
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=20 --lr_update=10 --learning_rate=.0005 --batch_size 256 \
#  --attn_name naaf --attn_smooth 15.0 --alpha 1.0


#GPU_DEVICES='1'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_naaf_vector_dp
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=20 --lr_update=10 --learning_rate=.0005 --batch_size 256 \
#  --attn_name naaf --attn_smooth 15.0 --alpha 1.0 --use_sim_metric


#GPU_DEVICES='2'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_naaf_matrix_dp_sim32
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=20 --lr_update=10 --learning_rate=.0005 --batch_size 256 \
#  --attn_name naaf --attn_smooth 15.0 --alpha 1.0 --use_sim_matrix --sim_dim 32