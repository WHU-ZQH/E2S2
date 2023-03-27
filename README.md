# Encoding-Enhanced Sequence-to-Sequence Pretraining

This is the official implementation of our paper, "[E2S2: Encoding-enhanced sequence-to-sequence pretraining for language understanding and generation](https://arxiv.org/pdf/2205.14912.pdf)" (in Pytorch).


## Requirements and Installation

- PyTorch version >= 1.10.0
- Python version >= 3.8
- For training, you'll also need an NVIDIA GPU and NCCL.
- To install **fairseq** and develop locally:

``` bash
git clone https://github.com/facebookresearch/fairseq.git
mv fairseq fairseq-setup
cd fairseq-setup
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./
```

## Getting Started
We integrate our E2S2 pretraining strategy in the fairseq script and provide the full code in "[fairseq-E2S2](https://github.com/WHU-ZQH/E2S2/tree/master/fairseq-E2S2)". Here, we introduce how to use our E2S2 to pretraining the BART model. 

### E2S2 for BART pretraining
To perform this process, you should first prepare the training environment by the following commands:

``` 
# removing the original scripts
rm -r fairseq-setup/fairseq
rm -r fairseq-setup/fairseq_cli/train.py

# using our self-questioning scripts
cp -r fairseq-sctd fairseq-setup/
mv fairseq-setup/fairseq-sctd fairseq-setup/fairseq
```

Then, you can follow the [pretraining README](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.pretraining.md) to prepare the pretraining data and model checkpoint.

Lastly, you can start E2S2 pretraining by the following commands:

``` 
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

```

## Evaluation
You can evaluate the pretrained models by using the original [fine-tuning scripts](https://github.com/facebookresearch/fairseq/tree/main/examples/bart), or any way you like.

## Citation
If you find this work helpful, please consider citing as follows:  

```ruby
@article{zhong2022e2s2,
  title={E2S2: Encoding-enhanced sequence-to-sequence pretraining for language understanding and generation},
  author={Zhong, Qihuang and Ding, Liang and Liu, Juhua and Du, Bo and Tao, Dacheng},
  journal={arXiv preprint arXiv:2205.14912},
  year={2022}
}
```
