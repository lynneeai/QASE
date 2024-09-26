import sys
sys.path.append("../..")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["multispanqa", "squad", "quoref"], default="multispanqa")
parser.add_argument("--batch", choices=["test", "val"], default="test")
parser.add_argument("--base_model", choices=["llama2", "alpaca", "flan-t5-small", "flan-t5-base", "flan-t5-large", "flan-t5-xl", "flan-t5-xxl"], default="llama2")
parser.add_argument("--ckpt_file", required=True)
args = parser.parse_args()

from configs.inference_qase_config import Inference_Config
# This needs to be initialized here to properly set up gpu devices
config = Inference_Config(args.dataset, args.batch, args.base_model)

import json
import torch
from tqdm import tqdm
from transformers import GenerationConfig

from qase import QASE
from configs.multispanqa.data_config import MultiSpanQA_Data_Config
from configs.squad.data_config import SQuAD_Data_Config
from configs.quoref.data_config import Quoref_Data_Config
from dataloaders.multispanqa_dataloader import MultiSpanQA_DataLoader
from dataloaders.squad_dataloader import SQuAD_DataLoader
from dataloaders.quoref_dataloader import Quoref_DataLoader
from inference.utils import (
    get_tagged_spans,
    postprocess_multispanqa_prediction,
    postprocess_squad_predictions,
    postprocess_quoref_predictions
)
from utils import set_llama2_credential


set_llama2_credential()


print(f"Start predicting on {args.dataset}...")
if args.dataset == "multispanqa":
    data_config = MultiSpanQA_Data_Config(config.batch_size, llama_based_model=config.llama_based_model, base_model=config.base_model)
    dataloader = MultiSpanQA_DataLoader(data_config)
    batch_dataloader = (
        dataloader.load_unlabeled_val_dataloader()
        if args.batch == "val"
        else dataloader.test_dataloader
    )
    batch_data_file = (
        "../../datasets/MultiSpanQA/valid_readable.json"
        if args.batch == "val"
        else "../../datasets/MultiSpanQA/test_readable.json"
    )
    batch_original_data = json.load(open(batch_data_file, "r"))

elif args.dataset == "squad":
    data_config = SQuAD_Data_Config(config.batch_size, llama_based_model=config.llama_based_model, base_model=config.base_model)
    dataloader = SQuAD_DataLoader(data_config)
    batch_dataloader = dataloader.test_dataloader
    batch_original_data = dataloader.test_dataset.to_list()

elif args.dataset == "quoref":
    data_config = Quoref_Data_Config(config.batch_size, llama_based_model=False, base_model=config.base_model)
    dataloader = Quoref_DataLoader(data_config)
    batch_dataloader = dataloader.test_dataloader
    batch_original_data = dataloader.test_dataset.to_list()

assert len(batch_dataloader) == len(batch_original_data)

# model setup
model = QASE(
    config=config,
    tokenizer=dataloader.tokenizer,
    inference=True
)
# TODO: Only support single GPU for now
print(f"Loading model weights from {args.ckpt_file}...")
model.load_state_dict(torch.load(args.ckpt_file), strict=False)
model.eval()

# inference
print("Performing inference...")
generation_config = GenerationConfig(max_new_tokens=32)
pred_objs = [] if args.dataset == "squad" else {}
# seq_tag_objs = []
for original_sample, sample in tqdm(
    list(zip(batch_original_data, list(batch_dataloader))),
    colour="green",
    dynamic_ncols=True
):
    sample.to(model.model.device)
    
    # generate
    if args.base_model in ["llama2", "alpaca"]:
        generated = model.model.generate(
            input_ids=sample["input_ids"],
            generation_config=generation_config
        )
        generated_text = dataloader.tokenizer.decode(generated[0])
        generated_response = dataloader.prompter.get_response(
            generated_text, eos_token=dataloader.tokenizer.eos_token
        )
    else:
        generated = model.model.generate(
            input_ids=sample["input_ids"],
            decoder_input_ids = torch.tensor(
                [dataloader.tokenizer.bos_token_id]
            ).unsqueeze(0).to(model.model.device),
            generation_config=generation_config
        )
        generated_text = dataloader.tokenizer.decode(generated[0])
        generated_response = generated_text.split(dataloader.tokenizer.eos_token)[0]
    if args.dataset == "multispanqa":
        pred_obj = postprocess_multispanqa_prediction(original_sample, generated_response)
        pred_objs = {**pred_objs, **pred_obj}
    elif args.dataset == "squad":
        pred_objs.append(postprocess_squad_predictions(original_sample, generated_response))
    elif args.dataset == "quoref":
        pred_obj = postprocess_quoref_predictions(original_sample, generated_response)
        pred_objs = {**pred_objs, **pred_obj}
    
    json.dump(
        pred_objs, open(config.predictions_output_file, "w"), indent=4
    )
    
    # # span tagging
    # with torch.no_grad():
    #     output_dict = model(sample)
    # tagged_spans = get_tagged_spans(sample, output_dict, dataloader.tokenizer)
    # seq_tag_objs.append({original_sample["id"]: tagged_spans})
    # json.dump(seq_tag_objs, open(config.seq_tag_output_file, "w"), indent=4)
    