#-------------------------------------------------------------------------
#----------------------- SCAN t2i by NAAF ----------------------------
#GPU_DEVICES='0'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=distill_${DATASET_NAME}_scan_t2i_by_naaf_use_cosine
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train_ensemble.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=15 --margin 0.2 --learning_rate=.0002 \
#  --attn_name scan --attn_type t2i --attn_smooth 9.0 \
#  --distill_name naaf --distill_type cosine --distill_factor 1.0
#
#GPU_DEVICES='0'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=distill_${DATASET_NAME}_scan_t2i_by_naaf_use_vector
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train_ensemble.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=15 --margin 0.2 --learning_rate=.0002 \
#  --attn_name scan --attn_type t2i --attn_smooth 9.0 \
#  --distill_name naaf --distill_type vector --distill_factor 1.0
#
#GPU_DEVICES='1'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=distill_${DATASET_NAME}_scan_t2i_by_naaf_use_matrix
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train_ensemble.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=30 --lr_update=15 --margin 0.2 --learning_rate=.0002 \
#  --attn_name scan --attn_type t2i --attn_smooth 9.0 \
#  --distill_name naaf --distill_type matrix --distill_factor 1.0

#----------------------------------------------------------------
#----------------------- VSE by VSRN ----------------------------
#GPU_DEVICES='2'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=distill_${DATASET_NAME}_vse_by_vsrn_use_cosine
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train_ensemble_vsrn.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=35 --lr_update=20 --margin 0.2 --learning_rate=.0002 \
#  --attn_name vse --attn_type cosine \
#  --distill_name vsrn --distill_type cosine --distill_factor 1e-3
#
#GPU_DEVICES='2'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=distill_${DATASET_NAME}_vse_by_vsrn_use_vector
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train_ensemble_vsrn.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=35 --lr_update=20 --margin 0.2 --learning_rate=.0002 \
#  --attn_name vse --attn_type cosine \
#  --distill_name vsrn --distill_type vector --distill_factor 1e-3
#
#GPU_DEVICES='2'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=distill_${DATASET_NAME}_vse_by_vsrn_use_matrix
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train_ensemble_vsrn.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=35 --lr_update=20 --margin 0.2 --learning_rate=.0002 \
#  --attn_name vse --attn_type cosine \
#  --distill_name vsrn --distill_type matrix --distill_factor 1e-3

#----------------------------------------------------------------
#----------------------- VSE by SCAN t2i ------------------------
#GPU_DEVICES='0'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=distill_${DATASET_NAME}_vse_by_scan_t2i_use_cosine
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train_ensemble.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=35 --lr_update=20 --margin 0.2 --learning_rate=.0002 \
#  --attn_name vse --attn_type cosine \
#  --distill_name scan_t2i --distill_type cosine --distill_factor 1.0
#
#GPU_DEVICES='2'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=distill_${DATASET_NAME}_vse_by_scan_t2i_use_vector
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train_ensemble.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=35 --lr_update=20 --margin 0.2 --learning_rate=.0002 \
#  --attn_name vse --attn_type cosine \
#  --distill_name scan_t2i --distill_type vector --distill_factor 1.0
#
#GPU_DEVICES='3'
#DATASET_NAME='f30k'
#DATA_PATH='./data/'${DATASET_NAME}
#VOCAB_PATH='./data/vocab'
#MODEL_NAME=distill_${DATASET_NAME}_vse_by_scan_t2i_use_matrix
#
#CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 train_ensemble.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH} \
#  --logger_name runs/${MODEL_NAME}/log --model_name runs/${MODEL_NAME} --gpuid ${GPU_DEVICES} \
#  --num_epochs=35 --lr_update=20 --margin 0.2 --learning_rate=.0002 \
#  --attn_name vse --attn_type cosine \
#  --distill_name scan_t2i --distill_type matrix --distill_factor 1.0

GPU=0
