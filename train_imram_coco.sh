#GPU_DEVICES='0,1,2,3'
#DATASET_NAME='coco'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_imram_baseline_dp
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=20 --lr_update=10 --margin 0.2 --learning_rate=.0005 \
#  --attn_name imram --attn_layer 3 --attn_smooth 11.0


#GPU_DEVICES='0,1,2,3'
#DATASET_NAME='coco'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_imram_vector_dp
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=20 --lr_update=10 --margin 0.2 --learning_rate=.0005 \
#  --attn_name imram --attn_layer 3 --attn_smooth 11.0 --use_sim_metric


#GPU_DEVICES='0,1,2,3'
#DATASET_NAME='coco'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_imram_matrix_dp_sim128
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=20 --lr_update=10 --margin 0.2 --learning_rate=.0005 \
#  --attn_name imram --attn_layer 3 --attn_smooth 11.0 --use_sim_matrix --sim_dim 128