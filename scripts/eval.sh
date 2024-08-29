CUDA_VISIBLE_DEVICES=0 \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=48275 eval_dvt.py \
                                    --run_name=nuscenes_seg \
                                    --project_root_dir=/home/dpc/djy/DVT \
                                    --seam_root_dir=/home/dpc/djy/Datasets/nuScenes_label \
                                    --dataset_root_dir=/home/dpc/djy/Datasets/nuScenes \
                                    --mode=test \
                                    --test_dataset=nuScenes \
                                    --resume=/home/dpc/djy/DVT/ckpt/dvt.pth \
                                    --config=nuscenes.ini \
