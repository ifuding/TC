#! /bin/bash
set -x

python3 main.py \
        --input-training-data-path ../data/pix72/ --output-model-path ../submit/ --debug True --train_verbose 1 \
        --input-previous-model-path ../submit/image_05108_10fold_zs_HFlip_02061/ \
        --input-previous-model-path ../data/model_sub/6_12_24_16_ini128_growth32_inistride1_augdata_05108_TF14_DensetNetProcess/ \
        --densenet_nfold 5 --densenet_batch_size 256 --densenet_epochs 1 --densenet_ensemble_nfold 1 --densenet_patience 30 \
        --dem_nfold 2 --dem_batch_size 128 --dem_epochs 2 --dem_ensemble_nfold 2 --dem_patience 10 --zs_model_type DEM_BC \
        --blocks 6,12,24,16 --weight_decay 1e-4 --kernel_initializer glorot_normal --aug_data False \
        --rotation_range 20 --shear_range 0.2 --zoom_range 0.2 --horizontal_flip True \
        --init_filters 128 --growth_rate 32 --reduction 0.5 --lr 1e-3 --init_stride 1 --img_flat_len 97 --cat_max 365 \
        --load_img_model False --predict_flat False --load_zs_model false --TTA 5 --neg_aug 2 --predict_prob False \
        --only_emb False --train_ft False --ft_model skipgram --ft_size 10 --pixel 72 \
        --ft_threads 12 --ft_iter 1