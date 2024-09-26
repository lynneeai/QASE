import sys
sys.path.append("..")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["multispanqa", "squad", "quoref"], default="multispanqa")
parser.add_argument("--data_portion", type=float, required=False)
parser.add_argument("--base_model", default="llama2", choices=["llama2", "alpaca"])
parser.add_argument("--checkpoint_steps", type=int, required=False)
parser.add_argument("--resume_from_ckpt_dir", type=str, required=False)
args = parser.parse_args()

# This needs to be added at the very top to set up gpu devices
from configs.finetune_llama_config import Finetune_Llama_Config
config = Finetune_Llama_Config(
    dataset=args.dataset, 
    base_model=args.base_model,
    save_checkpoint_after_steps=bool(args.checkpoint_steps)
)

import json
import time
import torch
from tqdm import tqdm
from torch import optim
from transformers import LlamaForCausalLM
from peft import PeftModel, LoraConfig, TaskType, prepare_model_for_int8_training, get_peft_model

from configs.multispanqa.data_config import MultiSpanQA_Data_Config
from configs.squad.data_config import SQuAD_Data_Config
from configs.quoref.data_config import Quoref_Data_Config
from dataloaders.multispanqa_dataloader import MultiSpanQA_DataLoader
from dataloaders.squad_dataloader import SQuAD_DataLoader
from dataloaders.quoref_dataloader import Quoref_DataLoader
from utils import set_llama2_credential


set_llama2_credential()

def setup_model(tokenizer, base_model_id, resume_ckpt_dir=None):
    print(f"Setting up {base_model_id} with LoRA...")
    model = LlamaForCausalLM.from_pretrained(
        base_model_id, load_in_8bit=True, device_map="auto"
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    model = prepare_model_for_int8_training(model)
    
    if resume_ckpt_dir:
        print(f"Loading LoRA weights from {resume_ckpt_dir}...")
        model = PeftModel.from_pretrained(model, resume_ckpt_dir, is_trainable=True)
    else:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            lora_dropout=0.05,
        )
        model = get_peft_model(model, peft_config)
    
    model.print_trainable_parameters()
    return model


def train(
    model, train_dataloader, val_dataloader, optimizer, lr_scheduler, train_config, checkpoint_steps=None
):
    train_loss = []
    train_ppl = []
    val_loss = []
    val_ppl = []
    epoch_times = []
    best_val_loss = float("inf")
    for epoch in range(train_config.num_epochs):
        epoch_start_time = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(
            colour="blue",
            desc=f"Training Epoch: {epoch + 1}",
            total=len(train_dataloader),
            dynamic_ncols=True,
        )
        for step, batch in enumerate(train_dataloader):
            outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
            loss = outputs.loss
            epoch_loss += loss.detach().float()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.update(1)
            pbar.set_description(
                f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float():.4f})"
            )
            
            if checkpoint_steps and step % checkpoint_steps == 0:
                print(f"Saving checkpoint at step {step}...")
                model.save_pretrained(train_config.step_ckpt_dir)
                print(f"Model is saved in {train_config.step_ckpt_dir}.")
        
        pbar.close()

        epoch_end_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_end_time)

        epoch_loss = epoch_loss / len(train_dataloader)
        epoch_ppl = torch.exp(epoch_loss)
        train_loss.append(epoch_loss)
        train_ppl.append(epoch_ppl)

        lr_scheduler.step()

        print(
            f"Epoch {epoch+1}: train_perplexity={epoch_ppl:.4f}, train_epoch_loss={epoch_loss:.4f}, epoch time {epoch_end_time}s"
        )

        # evaluate
        val_epoch_loss, val_epoch_ppl = evaluate(model, val_dataloader)
        print(f"Current val loss: {val_epoch_loss}; Best val loss: {best_val_loss}")
        if val_epoch_loss < best_val_loss:
            print("Saving new checkpoint...")
            model.save_pretrained(train_config.ckpt_dir)
            print(f"Model is saved in {train_config.ckpt_dir}.")
            best_val_loss = val_epoch_loss
            
            results = {"epoch": epoch + 1}
            results["epoch_train_loss"] = float(train_loss[-1].detach().cpu().numpy())
            results["epoch_train_ppl"] = float(train_ppl[-1].detach().cpu().numpy())
            results["epoch_val_loss"] = float(val_epoch_loss.detach().cpu().numpy())
            results["epoch_val_ppl"] = float(val_epoch_ppl.detach().cpu().numpy())
            results["epoch_time"] = epoch_times[-1]
            json.dump(results, open(train_config.results_output_file, "w"), indent=4)
            
        val_loss.append(val_epoch_loss)
        val_ppl.append(val_epoch_ppl)

    avg_train_loss = sum(train_loss) / len(train_loss)
    avg_train_ppl = sum(train_ppl) / len(train_ppl)
    avg_val_loss = sum(val_loss) / len(val_loss)
    avg_val_ppl = sum(val_ppl) / len(val_ppl)
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    results = {}
    results["avg_train_loss"] = float(avg_train_loss.detach().numpy())
    results["avg_train_ppl"] = float(avg_train_ppl.detach().numpy())
    results["avg_val_loss"] = float(avg_val_loss.detach().numpy())
    results["avg_val_prep"] = float(avg_val_ppl.detach().numpy())
    results["avg_epoch_time"] = avg_epoch_time

    return results


def evaluate(model, val_dataloader):
    model.eval()
    epoch_loss = 0.0
    for batch in tqdm(
        val_dataloader, colour="green", desc=f"Evaluating Epoch", dynamic_ncols=True
    ):
        with torch.no_grad():
            outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
            loss = outputs.loss
            epoch_loss += loss.detach().float()

    epoch_loss = epoch_loss / len(val_dataloader)
    return epoch_loss, torch.exp(epoch_loss)


if __name__ == "__main__":
    print(f"Start fine-tuning on {args.dataset}...")
    if args.dataset == "multispanqa":
        data_config = MultiSpanQA_Data_Config(config.batch_size, llama_based_model=True, base_model=args.base_model)
        dataloader = MultiSpanQA_DataLoader(data_config)
        train_dataloader=dataloader.train_dataloader
        val_dataloader=dataloader.val_dataloader
    elif args.dataset == "squad":
        data_config = SQuAD_Data_Config(config.batch_size, llama_based_model=True, base_model=args.base_model)
        dataloader = SQuAD_DataLoader(data_config)
        if args.data_portion:
            train_dataloader, val_dataloader = dataloader.loade_partial_train_val_dataloaders(args.data_portion)
        else:
            train_dataloader=dataloader.train_dataloader
            val_dataloader=dataloader.val_dataloader
    elif args.dataset == "quoref":
        data_config = Quoref_Data_Config(config.batch_size, llama_based_model=True, base_model=args.base_model)
        dataloader = Quoref_DataLoader(data_config)
        train_dataloader=dataloader.train_dataloader
        val_dataloader=dataloader.val_dataloader

    print(f"Start finetuning {config.base_model} on {args.dataset}.")
    model = setup_model(
        dataloader.tokenizer, 
        base_model_id=config.base_model_id, 
        resume_ckpt_dir=args.resume_from_ckpt_dir
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.gamma)
    results = train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        train_config=config,
        checkpoint_steps=args.checkpoint_steps if args.checkpoint_steps else None
    )
    print(f"=======Results=======\n{results}")
