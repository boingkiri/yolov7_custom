#!/bin/bash

device=0
weights=/mnt/home/jeongjun/layout_diffusion/yolov7/runs/train/0925_coco_only_seen_10percent/weights/best.pt
name=coco_cycle_1_likelihood
# data=data/0920_disc_guidance_time_dependent/time_dependent.yaml
source=/mnt/home/jeongjun/layout_diffusion/datasets/1016_vaal_nll/type_1
log_dir=nll_log
log_name=type_1
annotation=/mnt/home/datasets/manual/coco_2017/annotations/instances_train2017.json

# CUDA_VISIBLE_DEVICES=$device python calculate_likelihood.py --data $data --img 640 --batch 32 --conf 0.001 --iou 0.65 --device $device --weights $weights --name $name
CUDA_VISIBLE_DEVICES=$device python calculate_likelihood.py --source $source --device $device --weights $weights --name $name --log_dir=$log_dir --log_name=$log_name --annotation=$annotation