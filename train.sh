CUDA_VISIBLE_DEVICES=1 python eval.py --model=ts_cnn_mlp_baseline --dataset=traffic_sign
CUDA_VISIBLE_DEVICES=1 python eval.py --model=ts_cnn_x_mlp --dataset=traffic_sign
                       python eval.py --model=ts_cnn1att_mlp --dataset=traffic_sign
