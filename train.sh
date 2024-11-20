CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/dfine/custom/dfine_hgnetv2_n_custom.yml --use-amp --seed=0 -t dfine_n_coco.pth
