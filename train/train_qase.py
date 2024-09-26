import sys
sys.path.append("..")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["multispanqa", "squad", "quoref"], default="multispanqa")
parser.add_argument("--data_portion", type=float, required=False)
parser.add_argument("--base_model", choices=["llama2", "alpaca", "flan-t5-small", "flan-t5-base", "flan-t5-large", "flan-t5-xl", "flan-t5-xxl"], default="llama2")
parser.add_argument("--checkpoint_steps", type=int, required=False)
parser.add_argument("--resume_from_ckpt_file", type=str, required=False)
args = parser.parse_args()

from configs.train_qase_config import Train_Config
# This needs to be initialized here to properly set up gpu devices
config = Train_Config(
    args.dataset, 
    args.base_model,
    save_checkpoint_after_steps=bool(args.checkpoint_steps)
)

import torch
from configs.multispanqa.data_config import MultiSpanQA_Data_Config
from configs.squad.data_config import SQuAD_Data_Config
from configs.quoref.data_config import Quoref_Data_Config
from dataloaders.multispanqa_dataloader import MultiSpanQA_DataLoader
from dataloaders.squad_dataloader import SQuAD_DataLoader
from dataloaders.quoref_dataloader import Quoref_DataLoader
from qase import QASE
from utils import set_llama2_credential


set_llama2_credential()

print(f"Start training {args.base_model}_SpanQA_V2 on {args.dataset}...")
if args.dataset == "multispanqa":
    data_config = MultiSpanQA_Data_Config(config.batch_size, llama_based_model=config.llama_based_model, base_model=config.base_model)
    dataloader = MultiSpanQA_DataLoader(data_config)
    train_dataloader=dataloader.train_dataloader
    val_dataloader=dataloader.val_dataloader
elif args.dataset == "squad":
    data_config = SQuAD_Data_Config(config.batch_size, config.llama_based_model, config.base_model)
    dataloader = SQuAD_DataLoader(data_config)
    if args.data_portion:
        train_dataloader, val_dataloader = dataloader.loade_partial_train_val_dataloaders(args.data_portion)
    else:
        train_dataloader=dataloader.train_dataloader
        val_dataloader=dataloader.val_dataloader
elif args.dataset == "quoref":
    data_config = Quoref_Data_Config(config.batch_size, config.llama_based_model, config.base_model)
    dataloader = Quoref_DataLoader(data_config)
    train_dataloader=dataloader.train_dataloader
    val_dataloader=dataloader.val_dataloader

model = QASE(
    config=config,
    tokenizer=dataloader.tokenizer,
    span_labels_weight=dataloader.get_span_labels_weight()
)
if args.resume_from_ckpt_file:
    print(f"Loading checkpoint from {args.resume_from_ckpt_file}...")
    model.load_state_dict(torch.load(args.resume_from_ckpt_file), strict=False)
    
model.train_model(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    checkpoint_steps=args.checkpoint_steps if args.checkpoint_steps else None
)