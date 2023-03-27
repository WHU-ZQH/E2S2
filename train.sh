DATA_DIR=data-path
SAVE_DIR=save-path
MODEL_DIR=model-path

mkdir -p  $SAVE_DIR

fairseq-train $DATA_DIR \
    --train-subset train \
    --valid-subset valid \
    --skip-invalid-size-inputs-valid-test \
    --memory-efficient-fp16 \
    --fp16-init-scale 8 \
    --arch bart_large \
    --task denoising \
    --mask 0.15 `# Proportion to mask` \
    --mask-length "span-poisson" `# Mask a span of words, sampled with poisson distr lambda=3` \
    --replace-length 1 `# Replace spans of masked tokens with single <mask> token` \
    --permute-sentences 0.0 `# Paper states they permute all sentences` \
    --rotate 0.0 \
    --e2s2-shuffle 0.05 `# Proportion to shuffle` \
    --e2s2-random 0.05 `# Proportion to randomly replace` \
    --sample-break-mode "complete" \
    --max-tokens 4096 \
    --tokens-per-sample 1024 \
    --max-source-positions 1024 \
    --max-target-positions 1024 \
    --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 2 --no-epoch-checkpoints \
    --restore-file $MODEL_DIR \
    --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
    --optimizer adam --adam-betas "(0.9, 0.98)" \
    --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr 1.5e-6 \
    --lr-scheduler polynomial_decay \
    --warmup-updates 6000 \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --weight-decay 0.01 \
    --update-freq 3 \
    --ddp-backend=legacy_ddp \
    --total-num-update 50000 \
    --max-update 50000 \
    --save-dir $SAVE_DIR \
    --local_rank $SLURM_LOCALID \
    --log-format json --log-interval 100 2>&1 | tee $SAVE_DIR/train.log
