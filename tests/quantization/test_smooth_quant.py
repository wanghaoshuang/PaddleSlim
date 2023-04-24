# Copyright (c) 2023  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import unittest

sys.path.append("../../")
import numpy as np
import paddle
import tempfile
import json
from paddlenlp.transformers import AutoModelForConditionalGeneration
from paddleslim.quant.smooth.ln_smooth import PreLNSmooth
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.data import Stack, Tuple
from paddle.io import DataLoader


class Lambada_Eval_Dataset(paddle.io.Dataset):
    def __init__(self, tokens, labels, seq_len, pad_idx):
        self.seq_len = seq_len
        self.pad_idx = pad_idx
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.tokens)

    def _construct_sample(self, tokens):
        tokens = np.array(tokens).astype("int64").tolist()
        labels = tokens[1:]
        tokens = tokens[:-1]

        seq_length = len(tokens)
        # attention mask for the attention calulate
        attention_mask = np.tri(seq_length, seq_length).reshape((1, seq_length,
                                                                 seq_length))

        # the pad and eos tokens do not contribute the loss
        position_ids = np.arange(0, seq_length, dtype="int64")

        # -INF mask value as default
        # attention_mask = (attention_mask - 1.0) * 1e9
        # Bool mask of attention
        attention_mask = attention_mask.astype("float32")
        return [tokens, attention_mask, position_ids, labels]

    def __getitem__left_padding(self, idx):
        tokens = self.tokens[idx][:self.seq_len]
        labels = self.labels[idx]
        tokens = tokens + labels
        num_tokens = len(tokens)
        if num_tokens < self.seq_len + 1:
            num_pad = self.seq_len + 1 - num_tokens
            # tokens += [self.pad_idx] * num_pad + tokens
            tokens = [self.pad_idx] * num_pad + tokens
        loss_mask = np.zeros(self.seq_len, dtype="float32")
        loss_mask[-len(labels):] = 1.0
        [tokens, attention_mask, position_ids,
         labels] = self._construct_sample(tokens)
        return [tokens, loss_mask, attention_mask, position_ids, labels]

    def __getitem__(self, idx):
        tokens = self.tokens[idx][:self.seq_len]
        labels = self.labels[idx]
        tokens = tokens + labels
        num_tokens = len(tokens)
        if num_tokens < self.seq_len + 1:
            num_pad = self.seq_len + 1 - num_tokens
            tokens += [self.pad_idx] * num_pad
        loss_mask = np.zeros(self.seq_len, dtype="float32")
        loss_mask[num_tokens - len(labels) - 1:num_tokens - 1] = 1.0
        [tokens, attention_mask, position_ids,
         labels] = self._construct_sample(tokens)
        return [tokens, loss_mask, attention_mask, position_ids, labels]


def get_tokens(tokenizer, text, strict=True):
    if not strict:
        tokens = tokenizer(text)["input_ids"]
        return tokens[:-1], [tokens[-1]]
    last_token = text.split()[-1]
    start_idx = text.rfind(last_token)
    beginning_tokens = tokenizer(text[:start_idx].strip())["input_ids"]
    last_token = tokenizer(" " + last_token)["input_ids"]
    return beginning_tokens, last_token


class TestQuantAwareTraining(unittest.TestCase):
    def setUp(self):
        # paddle.set_device("cpu")
        self.dummy_input = paddle.rand([1, 3, 224, 224])
        self.temp_dir = tempfile.TemporaryDirectory(dir="./")
        self.path = os.path.join(self.temp_dir.name, 'smooth_quant')
        self.model_name = "THUDM/glm-large-chinese"

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_eval_dataset(self):
        val_dataloader = None
        eval_batch_size = 1
        seq_len = 1024

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenized_data = []
        tokenized_label = []
        eval_path = "./lambada_test.jsonl"
        with open(eval_path, "r") as f:
            for line in f.readlines():
                text = json.loads(line)["text"]
                tokens, labels = get_tokens(tokenizer, text)
                tokenized_data.append(tokens)
                tokenized_label.append(labels)
        val_dataset = Lambada_Eval_Dataset(tokenized_data, tokenized_label,
                                           seq_len, tokenizer.pad_token_id)
        num_tokenized_tokens = 0
        num_original_tokens = 0

        num_examples = len(val_dataset)
        num_original_tokens = num_original_tokens
        num_tokenized_tokens = num_tokenized_tokens
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            drop_last=False,
            collate_fn=Tuple(Stack(), Stack(), Stack(), Stack(), Stack()), )

        return val_dataloader

    def test_smooth(self):
        model = AutoModelForConditionalGeneration.from_pretrained(
            "THUDM/glm-large-chinese")
        # smooth = PreLNSmooth(model)
        smooth_step = 10
        custom_white_list = []
        eval_data_loader = self.create_eval_dataset()
        with paddle.no_grad():
            for step, batch in enumerate(eval_data_loader):
                tokens, loss_mask = batch[:2]
                with paddle.amp.auto_cast(False, level="O2", dtype="float16"):
                    pred = model(tokens)
                print(pred)
                break
                # if step == smooth_step:
                #     smooth.update_weight()
                #     break


if __name__ == '__main__':
    unittest.main()
