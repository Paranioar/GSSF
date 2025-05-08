#----------------------- T2I poly ----------------------------
#GPU_DEVICES='0'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_scan_baseline_t2i_dp_poly
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=15 --margin 0.2 --learning_rate=.0002 \
#  --attn_name scan --attn_type t2i --attn_smooth 9.0 \
#  --loss_name poly
#
#GPU_DEVICES='0'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_scan_vector_t2i_dp_poly
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=15 --margin 0.2 --learning_rate=.0002 \
#  --attn_name scan --attn_type t2i --attn_smooth 9.0 --use_sim_metric \
#    --loss_name poly
#
#GPU_DEVICES='0'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_scan_matrix_t2i_dp_sim256_poly
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=15 --margin 0.2 --learning_rate=.0002 \
#  --attn_name scan --attn_type t2i --attn_smooth 9.0 --use_sim_matrix --sim_dim 256 \
#  --loss_name poly

#----------------------- T2I cmpm ----------------------------
#GPU_DEVICES='1'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_scan_baseline_t2i_dp_cmpm
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=15 --margin 0.2 --learning_rate=.0002 \
#  --attn_name scan --attn_type t2i --attn_smooth 9.0 \
#  --loss_name cmpm --loss_smooth 20.0
#
#GPU_DEVICES='1'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_scan_vector_t2i_dp_cmpm
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=15 --margin 0.2 --learning_rate=.0002 \
#  --attn_name scan --attn_type t2i --attn_smooth 9.0 --use_sim_metric \
#    --loss_name cmpm --loss_smooth 20.0
#
#GPU_DEVICES='1'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_scan_matrix_t2i_dp_sim256_cmpm
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=15 --margin 0.2 --learning_rate=.0002 \
#  --attn_name scan --attn_type t2i --attn_smooth 9.0 --use_sim_matrix --sim_dim 256 \
#  --loss_name cmpm --loss_smooth 20.0

#----------------------- T2I infonce ----------------------------
#GPU_DEVICES='2'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_scan_baseline_t2i_dp_infonce
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=15 --margin 0.2 --learning_rate=.0002 \
#  --attn_name scan --attn_type t2i --attn_smooth 9.0 \
#  --loss_name infonce --loss_smooth 20.0
#
#GPU_DEVICES='2'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_scan_vector_t2i_dp_infonce
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=15 --margin 0.2 --learning_rate=.0002 \
#  --attn_name scan --attn_type t2i --attn_smooth 9.0 --use_sim_metric \
#    --loss_name infonce --loss_smooth 20.0
#
#GPU_DEVICES='2'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_scan_matrix_t2i_dp_sim256_infonce
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=15 --margin 0.2 --learning_rate=.0002 \
#  --attn_name scan --attn_type t2i --attn_smooth 9.0 --use_sim_matrix --sim_dim 256 \
#  --loss_name infonce --loss_smooth 20.0

#-------------------------------------------------------------
#----------------------- I2T poly ----------------------------
#GPU_DEVICES='2'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_scan_baseline_i2t_dp_poly
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=15 --margin 0.2 --learning_rate=.0002 \
#  --attn_name scan --attn_type i2t --attn_smooth 4.0 \
#  --loss_name poly
#
#GPU_DEVICES='2'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_scan_vector_i2t_dp_poly
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=15 --margin 0.2 --learning_rate=.0002 \
#  --attn_name scan --attn_type i2t --attn_smooth 4.0 --use_sim_metric \
#  --loss_name poly
#
#GPU_DEVICES='2'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_scan_matrix_i2t_dp_sim256_poly
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=15 --margin 0.2 --learning_rate=.0002 \
#  --attn_name scan --attn_type i2t --attn_smooth 4.0 --use_sim_matrix --sim_dim 256 \
#  --loss_name poly

#----------------------- I2T cmpm ----------------------------
#GPU_DEVICES='1'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_scan_baseline_i2t_dp_cmpm
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=15 --margin 0.2 --learning_rate=.0002 \
#  --attn_name scan --attn_type i2t --attn_smooth 4.0 \
#  --loss_name cmpm --loss_smooth 20.0
#
#GPU_DEVICES='1'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_scan_vector_i2t_dp_cmpm
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=15 --margin 0.2 --learning_rate=.0002 \
#  --attn_name scan --attn_type i2t --attn_smooth 4.0 --use_sim_metric \
#  --loss_name cmpm --loss_smooth 20.0
#
#GPU_DEVICES='1'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_scan_matrix_i2t_dp_sim256_cmpm
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=15 --margin 0.2 --learning_rate=.0002 \
#  --attn_name scan --attn_type i2t --attn_smooth 4.0 --use_sim_matrix --sim_dim 256 \
#  --loss_name cmpm --loss_smooth 20.0

#----------------------- I2T infonce ----------------------------
#GPU_DEVICES='0'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_scan_baseline_i2t_dp_infonce
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=15 --margin 0.2 --learning_rate=.0002 \
#  --attn_name scan --attn_type i2t --attn_smooth 4.0 \
#  --loss_name infonce --loss_smooth 20.0
#
#GPU_DEVICES='0'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_scan_vector_i2t_dp_infonce
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=15 --margin 0.2 --learning_rate=.0002 \
#  --attn_name scan --attn_type i2t --attn_smooth 4.0 --use_sim_metric \
#  --loss_name infonce --loss_smooth 20.0
#
#GPU_DEVICES='0'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=${DATASET_NAME}_scan_matrix_i2t_dp_sim256_infonce
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=15 --margin 0.2 --learning_rate=.0002 \
#  --attn_name scan --attn_type i2t --attn_smooth 4.0 --use_sim_matrix --sim_dim 256 \
#  --loss_name infonce --loss_smooth 20.0

GPU=0