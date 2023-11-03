#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

fairseq-train data/cl_data/bin \
    --save-dir model/finetune_cl_with_mgenre \
    --user-dir custom_model \
    --restore-file model/finetune_lang_type/model.pt \
    --arch mbart_large_cl  \
    --task translation  \
    --criterion label_smoothed_cross_entropy_cl  \
    --source-lang source  \
    --target-lang target  \
    --truncate-source  \
    --label-smoothing 0.1  \
    --max-tokens 1024  \
    --update-freq 1  \
    --max-update 2000  \
    --required-batch-size-multiple 1  \
    --dropout 0.1  \
    --attention-dropout 0.1  \
    --relu-dropout 0.0  \
    --weight-decay 0.01  \
    --optimizer adam  \
    --adam-betas "(0.9, 0.999)"  \
    --adam-eps 1e-08  \
    --clip-norm 0.1  \
    --lr-scheduler polynomial_decay  \
    --lr 3e-06  \
    --total-num-update 200000  \
    --warmup-updates 500  \
    --ddp-backend no_c10d  \
    --num-workers 1  \
    --reset-meters  \
    --reset-optimizer \
    --share-all-embeddings \
    --layernorm-embedding \
    --share-decoder-input-output-embed  \
    --skip-invalid-size-inputs-valid-test  \
    --log-format json  \
    --log-interval 10  \
    --patience 200 \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --amp \
    --save-interval-updates 500
