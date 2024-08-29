
CUDA_VISIBLE_DEVICES=0 \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=48275 train_dvt.py \
                                    --run_name=nuscenes \
                                    --project_root_dir=/home/dpc/djy/DVT \
                                    --seam_root_dir=/home/dpc/djy/Datasets/nuScenes_label \
                                    --dataset_root_dir=/home/dpc/djy/Datasets/nuScenes \
                                    --mode=train \
                                    --train_dataset=nuScenes \
                                    --val_dataset=nuScenes \
                                    --config=nuscenes.ini
