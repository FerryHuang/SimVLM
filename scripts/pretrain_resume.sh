#! bin/bash

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    main_pretrain.py \
    --distributed True \
    --batch_size 16 \
    --epochs 5 \
    --save_every 1 \
    --accum_iter 1 \
    --warmup_steps 2000 \
    --num_workers 8 \
    --run_name 'on_coco' \
    --resume 'must' \
    --run_id '2ohshqga' \
    --checkpoint_path 'result/run-2022-12-17 21:10:54/-checkpoint-epoch2.pth'