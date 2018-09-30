#! /bin/bash
set -x

python3 main.py \
        --input-training-data-path ../../Data/ --output-model-path ../submit/ --debug True --train_verbose 2 \
        --input-previous-model-path ../../Data/model_sub/6_12_24_16_ini64_growth32_inistride1_augdata_05010/ \
        --densenet_nfold 5 --densenet_batch_size 256 --densenet_epochs 1 --densenet_ensemble_nfold 1 --densenet_patience 30 \
        --dem_nfold 5 --dem_batch_size 64 --dem_epochs 2 --dem_ensemble_nfold 2 --dem_patience 10 \
        --blocks 16,16,16,16 --weight_decay 1e-4 --kernel_initializer glorot_normal --aug_data True \
        --init_filters 64 --growth_rate 32 --reduction 0.6 --lr 1e-3 --init_stride 1 --img_flat_len 1024 --cat_max 365 \
        --load_img_model False