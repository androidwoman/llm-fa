python3.10 main2.py \
    --model_path=HooshvareLab/gpt2-fa \
    --output_dir=results \
    --per_device_train_batch_size=2 \
    --learning_rate=3e-5 \
    --num_train_epochs=2 \
    --optim=adamw_torch \
    --r=8 \
    --lora_alpha=32\
    --lora_dropout=0.05\

python3.10 run_model.py