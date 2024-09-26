import copy
import json
import torch
from datasets import Dataset
from transformers import LlamaTokenizerFast, AutoTokenizer, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from utils import find_all_substrings, find_all_sublists
from prompters.multispanqa_prompter import MultiSpanQA_Prompter


class MultiSpanQA_DataLoader(object):
    def __init__(self, data_config, prompter_instruction=None):
        print("Initializing MultiSpanQA dataset...")
        self.data_config = data_config

        self._setup_tokenizer()
        self.prompter = MultiSpanQA_Prompter(instruction=prompter_instruction)

        self.type_dict = {
            "DESC": "description",
            "ENTY": "entity",
            "HUM": "human",
            "LOC": "location",
            "NUM": "numeric",
        }
        
        self.question_sep_idx_dict = {
            "llama2": [13, 5634, 13, 16492, 29901], # "\n---\nQuestion:"
            "alpaca": [13, 5634, 13, 16492, 29901], # "\n---\nQuestion:"
            "flan-t5-small": [14817, 11860, 10], # "--- Question:"
            "flan-t5-base": [14817, 11860, 10], # "--- Question:"
            "flan-t5-large": [14817, 11860, 10], # "--- Question:"
            "flan-t5-xl": [14817, 11860, 10], # "--- Question:"
            "flan-t5-xxl": [14817, 11860, 10], # "--- Question:"
        }
        self.context_sep_idx_dict = {
            "llama2": [13, 5634, 13, 2677, 29901], # "\n---\nContext:"
            "alpaca": [13, 5634, 13, 2677, 29901], # "\n---\nContext:"
            "flan-t5-small": [14817, 1193, 6327, 10], # "--- Context:"
            "flan-t5-base": [14817, 1193, 6327, 10], # "--- Context:"
            "flan-t5-large": [14817, 1193, 6327, 10], # "--- Context:"
            "flan-t5-xl": [14817, 1193, 6327, 10], # "--- Context:"
            "flan-t5-xxl": [14817, 1193, 6327, 10], # "--- Context:"
        }
        self.answer_sep_idx_dict = {
            "llama2": [13, 5634, 13, 22550, 29901], # "\n---\nAnswer:"
            "alpaca": [13, 5634, 13, 22550, 29901], # "\n---\nAnswer:"
            "flan-t5-small": [14817, 11801, 10], # "--- Answer:"
            "flan-t5-base": [14817, 11801, 10], # "--- Answer:"
            "flan-t5-large": [14817, 11801, 10], # "--- Answer:"
            "flan-t5-xl": [14817, 11801, 10], # "--- Answer:"
            "flan-t5-xxl": [14817, 11801, 10], # "--- Answer:"
        }

    def _setup_tokenizer(self):
        if self.data_config.llama_based_model:
            print("Setting up tokenizer from meta-llama/Llama-2-7b-hf")
            self.tokenizer = LlamaTokenizerFast.from_pretrained("meta-llama/Llama-2-7b-hf")
            self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        else:
            assert bool(self.data_config.base_model_id)
            print(f"Setting up tokenizer from {self.data_config.base_model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.data_config.base_model_id)
            self.tokenizer.add_special_tokens({"bos_token": "<s>"})

    def _read_datapoint(self, dp, load_labels=True):
        processed_dp = {
            "type": self.type_dict[dp["type"]] if load_labels else None,
            "question": " ".join(dp["question"]).capitalize() + " ?",
            "context": " ".join(dp["context"]),
            "spans": [] if load_labels else None,
        }

        if load_labels:
            cur_span = []
            for tok, l in zip(dp["context"], dp["label"]):
                if l == "B" or l == "I":
                    cur_span.append(tok)
                else:
                    if cur_span:
                        processed_dp["spans"].append(" ".join(cur_span))
                        cur_span = []

            processed_dp["spans_indices"] = [
                find_all_substrings(processed_dp["context"], s)[0]
                for s in processed_dp["spans"]
            ]
        return processed_dp

    def _get_question_indices(self, input_ids):
        question_sep_idx = find_all_sublists(
            input_ids, self.question_sep_idx_dict[self.data_config.base_model]
        )[0]
        answer_sep_idx = find_all_sublists(
            input_ids, self.answer_sep_idx_dict[self.data_config.base_model]
        )[0]
        return [question_sep_idx[1], answer_sep_idx[0]]

    def _get_context_indices(self, input_ids):
        context_sep_idx = find_all_sublists(
            input_ids, self.context_sep_idx_dict[self.data_config.base_model]
        )[0]
        question_sep_idx = find_all_sublists(
            input_ids, self.question_sep_idx_dict[self.data_config.base_model]
        )[0]
        return [context_sep_idx[1], question_sep_idx[0]]

    def _get_span_labels(self, spans_indices, offset_mapping, context_indices):
        def _get_start_end_positions(start, end, offset_mapping):
            for i in range(len(offset_mapping)):
                if int(offset_mapping[i][0]) <= start < int(offset_mapping[i][1]):
                    start_idx = i
                    break
            for i in range(len(offset_mapping) - 1, -1, -1):
                if int(offset_mapping[i][0]) <= end <= int(offset_mapping[i][1]):
                    if end == int(offset_mapping[i][1]):
                        end_idx = i + 1
                    else:
                        end_idx = i
                    break
            return start_idx, end_idx

        span_labels = [0] * len(offset_mapping)
        for s_indices in spans_indices:
            start_idx, end_idx = _get_start_end_positions(
                s_indices[0], s_indices[1], offset_mapping
            )
            for i in range(start_idx, end_idx):
                span_labels[i] = 1
        return span_labels[context_indices[0] : context_indices[1]]

    def _tokenize(self, dp, load_labels=True):
        dp = self._read_datapoint(dp, load_labels=load_labels)
        prompt = self.prompter.generate_prompt(
            question=dp["question"],
            context=dp["context"],
            spans=" ; ".join(dp["spans"]) if load_labels and self.data_config.llama_based_model else None,
        )
        if load_labels and self.data_config.llama_based_model:
            prompt += self.tokenizer.eos_token
        prompt_tok = self.tokenizer(prompt, return_offsets_mapping=True)
        input_ids = prompt_tok["input_ids"]

        question_indices = self._get_question_indices(input_ids)
        context_indices = self._get_context_indices(input_ids)

        # labels
        if load_labels:
            if self.data_config.llama_based_model:
                labels = copy.deepcopy(input_ids)
                answer_sep_idx = find_all_sublists(
                    input_ids, self.answer_sep_idx_dict[self.data_config.base_model]
                )[0]
                for i in range(answer_sep_idx[1]):
                    labels[i] = -100  # The default setting for training with mask
            else:
                labels = self.tokenizer(self.tokenizer.bos_token + " ; ".join(dp["spans"]))["input_ids"]

            # spans labels
            span_indices_offset = len(prompt.split(dp["context"])[0])
            spans_indices = [
                (item[0] + span_indices_offset, item[1] + span_indices_offset)
                for item in dp["spans_indices"]
            ]
            span_labels = self._get_span_labels(
                spans_indices, prompt_tok["offset_mapping"], context_indices
            )

            return {
                **prompt_tok,
                **{
                    "question_indices": question_indices,
                    "context_indices": context_indices,
                    "labels": labels,
                    "span_labels": span_labels,
                },
            }

        else:
            return {
                **prompt_tok,
                **{
                    "question_indices": question_indices,
                    "context_indices": context_indices,
                },
            }

    def _collate_fn(self, batch, load_labels=True):
        seq2seq_collate = DataCollatorForSeq2Seq(
            self.tokenizer, padding=True, return_tensors="pt"
        )

        if load_labels:
            span_labels = []
            for feat in batch:
                span_labels.append(torch.LongTensor(feat["span_labels"]))
                del feat["span_labels"]
            padded_features = seq2seq_collate(batch)
            padded_span_labels = pad_sequence(span_labels, batch_first=True)
            return {**padded_features, **{"span_labels": padded_span_labels}}

        padded_features = seq2seq_collate(batch)
        return padded_features

    def _load_data(self, data_file, shuffle=True, load_labels=True):
        data = json.load(open(data_file, "r"))["data"]
        dataset = Dataset.from_list(data)
        dataset = dataset.map(lambda x: self._tokenize(x, load_labels=load_labels))
        remove_columns = [
            col
            for col in dataset.features.keys()
            if col
            not in [
                "input_ids",
                "attention_mask",
                "question_indices",
                "context_indices",
                "labels",
                "span_labels",
            ]
        ]
        dataset = dataset.remove_columns(remove_columns)
        dataloader = DataLoader(
            dataset,
            batch_size=self.data_config.batch_size,
            shuffle=shuffle,
            collate_fn=(lambda x: self._collate_fn(x, load_labels=load_labels)),
        )
        return dataloader

    def load_unlabeled_val_dataloader(self):
        return self._load_data(
            self.data_config.val_file, shuffle=False, load_labels=False
        )

    def get_span_labels_weight(self):
        train_data = json.load(open(self.data_config.train_file, "r"))["data"]
        sentence_lens, span_lens = [], []
        for sample in train_data:
            label = sample["label"]
            sentence_lens.append(len(label))
            span = [1 for l in label if l != "O"]
            span_lens.append(sum(span))

        total_toks = sum(sentence_lens)
        total_spans = sum(span_lens)
        nonspan_weight = total_toks / (total_toks - total_spans)
        span_weight = total_toks / total_spans
        return torch.Tensor([nonspan_weight, span_weight])

    @property
    def train_dataloader(self):
        return self._load_data(self.data_config.train_file)

    @property
    def val_dataloader(self):
        return self._load_data(self.data_config.val_file)

    @property
    def test_dataloader(self):
        return self._load_data(
            self.data_config.test_file, shuffle=False, load_labels=False
        )
