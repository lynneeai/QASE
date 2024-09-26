import sys
sys.path.append("../..")

import argparse
import os
import backoff
import openai
import json
from dotenv import load_dotenv
from tqdm import tqdm

from configs.multispanqa.data_config import MultiSpanQA_Data_Config
from configs.squad.data_config import SQuAD_Data_Config
from configs.quoref.data_config import Quoref_Data_Config
from dataloaders.multispanqa_dataloader import MultiSpanQA_DataLoader
from dataloaders.squad_dataloader import SQuAD_DataLoader
from dataloaders.quoref_dataloader import Quoref_DataLoader
from utils import longest_overlapping_substring
from inference.utils import postprocess_squad_predictions, postprocess_quoref_predictions


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


@backoff.on_exception(
    backoff.expo,
    (
        openai.error.RateLimitError,
        openai.error.APIError,
        openai.error.TryAgain,
        openai.error.Timeout,
        openai.error.APIConnectionError,
        openai.error.ServiceUnavailableError,
    ),
    max_tries=10,
)
def call_gpt(prompt, system_message="You are a helpful assistant.", model="gpt-4"):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    
    response = openai.ChatCompletion.create(model=model, messages=messages)
    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["multispanqa", "squad", "quoref"], default="multispanqa")
    parser.add_argument("--batch", choices=["test", "val"], default="test")
    parser.add_argument("--model", choices=["gpt-3.5-turbo", "gpt-4"], default="gpt-3.5-turbo")
    args = parser.parse_args()
    
    output_dir = f"./{args.dataset}_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    predictions_output_file = f"{output_dir}/{args.model}_{args.batch}_predictions.json"
    
    print(f"Start predicting on {args.dataset} with {args.model}...")
    if args.dataset == "multispanqa":
        data_config = MultiSpanQA_Data_Config(1, llama_based_model=True)
        dataloader = MultiSpanQA_DataLoader(
            data_config=data_config,
            prompter_instruction=(
                "Using the provided context, answer the question with exact phrases and avoid explanations. "
                "Format the response as follows: [\"answer1\", \"answer2\", ...]."
            )
        )
        print(dataloader.prompter.instruction)
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
        data_config = SQuAD_Data_Config(1, llama_based_model=True)
        dataloader = SQuAD_DataLoader(data_config)
        batch_dataloader = dataloader.test_dataloader
        batch_original_data = dataloader.test_dataset.to_list()
    
    elif args.dataset == "quoref":
        data_config = Quoref_Data_Config(1, llama_based_model=True)
        dataloader = Quoref_DataLoader(data_config)
        batch_dataloader = dataloader.test_dataloader
        batch_original_data = dataloader.test_dataset.to_list()
    
    # inference
    pred_objs = [] if args.dataset == "squad" else {}
    for original_sample, sample in tqdm(
        list(zip(batch_original_data, list(batch_dataloader))),
        colour="green",
        dynamic_ncols=True
    ):
        prompt = dataloader.tokenizer.decode(sample["input_ids"][0])
        response = call_gpt(prompt, model=args.model)
        
        if args.dataset == "multispanqa":
            try:
                spans = json.loads(response)
                assert isinstance(spans, list)
                spans = [
                    longest_overlapping_substring(original_sample["context"], s).strip()
                    for s in spans
                ]
            except json.JSONDecodeError:
                spans = [
                    longest_overlapping_substring(original_sample["context"], response).strip()
                ]
            pred_objs[original_sample["id"]] = [s for s in spans if s]
        
        elif args.dataset == "squad":
            pred_objs.append(postprocess_squad_predictions(original_sample, response))
        
        elif args.dataset == "quoref":
            processed_obj = postprocess_quoref_predictions(original_sample, response)
            pred_objs = {**pred_objs, **processed_obj}
        
        json.dump(pred_objs, open(predictions_output_file, "w"), indent=4)
            
            