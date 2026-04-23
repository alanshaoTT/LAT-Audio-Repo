MAX_PIXELS=1003520 \
swift infer \
    --model /path/to/lat-audio-base \
    --val_dataset /testset/path \
    --stream true \
    --load_data_args false \
    --max_new_tokens 10000 \
    --result_path /result/path \
