# python 0split_dir.py
python 1pp_from_video.py
python 2pp_cpu.py
CUDA_VISIBLE_DEVICES=0 python 3pp_gpu.py --start_idx 0 --pp_num 100 &
CUDA_VISIBLE_DEVICES=1 python 3pp_gpu.py --start_idx 100 --pp_num 100 &
CUDA_VISIBLE_DEVICES=2 python 3pp_gpu.py --start_idx 200 --pp_num 100 &
CUDA_VISIBLE_DEVICES=3 python 3pp_gpu.py --start_idx 300 --pp_num 100