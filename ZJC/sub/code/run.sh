#! /bin/bash
set -x

python3 main.py \
        --input-training-data-path ../data/ --output-model-path ../submit/ --debug True --train_verbose 1 \
        --input-previous-model-path ../data/model_sub/6_12_24_16_ini128_growth32_inistride1_augdata_04901_Round1_Round2A/ \
        --densenet_nfold 5 --densenet_batch_size 256 --densenet_epochs 1 --densenet_ensemble_nfold 1 --densenet_patience 30 \
        --dem_nfold 2 --dem_batch_size 256 --dem_epochs 2 --dem_ensemble_nfold 2 --dem_patience 10 --zs_model_type DEM \
        --blocks 6,12,24,16 --weight_decay 1e-4 --kernel_initializer glorot_normal --aug_data True \
        --rotation_range 0 --shear_range 0 --zoom_range 0 --horizontal_flip False \
        --init_filters 128 --growth_rate 32 --reduction 0.5 --lr 1e-3 --init_stride 1 --img_flat_len 1032 --cat_max 365 \
        --load_img_model True --predict_flat False  