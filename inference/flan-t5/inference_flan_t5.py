import sys
sys.path.append("../..")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["multispanqa", "squad", "quoref"], default="multispanqa")
parser.add_argument("--batch", choices=["test", "val"], default="test")
parser.add_argument("--base_model", default="flan-t5-small", choices=["flan-t5-small", "flan-t5-base", "flan-t5-large", "flan-t5-xl", "flan-t5-xxl"])
parser.add_argument("--ckpt_dir", default=None)
args = parser.parse_args()


from configs.inference_flan_t5_config import Inference_Flan_T5_Config
# This needs to be initialized here to properly set up gpu devices
config = Inference_Flan_T5_Config(args.dataset, args.batch, args.base_model, ckpt_dir=args.ckpt_dir)

import json
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, GenerationConfig
from configs.multispanqa.data_config import MultiSpanQA_Data_Config
from configs.squad.data_config import SQuAD_Data_Config
from configs.quoref.data_config import Quoref_Data_Config
from dataloaders.multispanqa_dataloader import MultiSpanQA_DataLoader
from dataloaders.squad_dataloader import SQuAD_DataLoader
from dataloaders.quoref_dataloader import Quoref_DataLoader
from inference.utils import (
    postprocess_multispanqa_prediction,
    postprocess_squad_predictions,
    postprocess_quoref_predictions
)


print(f"Start predicting on {args.dataset}...")
if args.dataset == "multispanqa":
    data_config = MultiSpanQA_Data_Config(config.batch_size, llama_based_model=False, base_model=config.base_model)
    if not args.ckpt_dir:
        prompter_instruction = (
            "Using the provided context, answer the question with exact phrases and avoid explanations. "
            "Format the response as follows: [\"answer1\", \"answer2\", ...]."
        )
    else:
        prompter_instruction = None
    print(f"prompter_instruction: {prompter_instruction}")
    dataloader = MultiSpanQA_DataLoader(data_config, prompter_instruction=prompter_instruction)
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
    data_config = SQuAD_Data_Config(config.batch_size, llama_based_model=False, base_model=config.base_model)
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
print(f"Setting up {args.base_model}...")
if args.ckpt_dir:
    print(f"Loading fine-tuned {args.base_model} from {args.ckpt_dir}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.ckpt_dir, device_map="auto")
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_id, device_map="auto")
model.resize_token_embeddings(len(dataloader.tokenizer))
model.config.bos_token_id = dataloader.tokenizer.bos_token_id
model.eval()

# inference
print("Performing inference...")
generation_config = GenerationConfig(max_new_tokens=32)
generated_objs = [] if args.dataset == "squad" else {}
for original_sample, sample in tqdm(
    list(zip(batch_original_data, list(batch_dataloader))),
    colour="green",
    dynamic_ncols=True
):
    generated = model.generate(
        input_ids=sample["input_ids"].to(model.device),
        decoder_input_ids = torch.tensor(
            [dataloader.tokenizer.bos_token_id]
        ).unsqueeze(0).to(model.device),
        generation_config=generation_config
    )
    generated_text = dataloader.tokenizer.decode(generated[0])
    generated_response = generated_text.split(dataloader.tokenizer.eos_token)[0]
    if args.dataset == "multispanqa":
        processed_obj = postprocess_multispanqa_prediction(original_sample, generated_response)
        generated_objs = {**generated_objs, **processed_obj}
    elif args.dataset == "squad":
        generated_objs.append(postprocess_squad_predictions(original_sample, generated_response))
    elif args.dataset == "quoref":
        processed_obj = postprocess_quoref_predictions(original_sample, generated_response)
        generated_objs = {**generated_objs, **processed_obj}
    json.dump(
        generated_objs, open(config.predictions_output_file, "w"), indent=4
    )