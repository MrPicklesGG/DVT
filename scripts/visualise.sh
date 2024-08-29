
CUDA_VISIBLE_DEVICES=0 \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=48275 visualise.py \
                                    --run_name=vis_0 \
                                    --project_root_dir=/home/dpc/djy/DVT \
                                    --seam_root_dir=/home/dpc/djy/Datasets/nuScenes_panopticbev\
                                    --dataset_root_dir=/home/dpc/djy/Datasets/nuScenes \
                                    --mode=test \
                                    --test_dataset=nuScenes \
                                    --resume=/home/dpc/djy/DVT/ckpt/dvt.pth \
                                    --config=nuscenes.ini \
