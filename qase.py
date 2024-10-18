import sys

sys.path.append("..")

import time
import torch
import json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch import optim
from transformers import LlamaForCausalLM, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import Seq2SeqLMOutput
from torch.nn import MultiheadAttention
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.classification import BinaryAccuracy
from peft import LoraConfig, TaskType, prepare_model_for_int8_training, get_peft_model


class QASE(nn.Module):
    def __init__(self, config, tokenizer, span_labels_weight=None, inference=False):
        super(QASE, self).__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.inference = inference

        self._setup_model()

        # text output embedding
        self.text_output_layer = nn.Linear(
            self.model.config.hidden_size, config.text_output_layer_dim
        ).to(self.model.device)
        # span selection layers
        self.span_attn = MultiheadAttention(
            config.text_output_layer_dim, config.num_heads
        ).to(self.model.device)
        self.span_output_layer = nn.Linear(config.text_output_layer_dim, 2).to(self.model.device)

        if inference:
            self.text_output_layer.to(torch.float16)
            self.span_attn.to(torch.float16)
            self.span_output_layer.to(torch.float16)

        # loss
        self.celoss = nn.CrossEntropyLoss(weight=span_labels_weight).to(self.model.device)
        self.binary_accuracy = BinaryAccuracy().to(self.model.device)

    def _setup_model(self):
        if self.config.base_model in ["llama2", "alpaca"]:
            print(f"Setting up {self.config.base_model} with LoRA...")
            self.model = LlamaForCausalLM.from_pretrained(
                self.config.base_model_id, load_in_8bit=True, device_map="auto"
            )
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

            if not self.inference:
                self.model = prepare_model_for_int8_training(self.model)

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=self.inference,
                r=8,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                lora_dropout=0.05,
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        else:
            print(f"Setting up {self.config.base_model}...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.base_model_id, device_map="auto"
            )
            self.model.resize_token_embeddings(len(self.tokenizer))
            if self.inference:
                self.model.eval()
            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Trainable parameters: {num_params}")

    def forward(self, x):
        for key, val in x.items():
            x[key] = val.to(self.model.device)
        
        outputs = self.model(
            input_ids=x["input_ids"],
            attention_mask=x["attention_mask"],
            labels=x["labels"] if not self.inference else None,
            output_hidden_states=True,
        )
        lm_loss = outputs.loss if not self.inference else None
        if isinstance(outputs, Seq2SeqLMOutput):
            outputs_embed = outputs.encoder_last_hidden_state
        else:
            outputs_embed = outputs.hidden_states[0]  # [batch_size, seq_len, 4096]
        outputs_embed = F.leaky_relu(
            self.text_output_layer(outputs_embed)
        )  # [batch_size, seq_len, text_output_layer_dim]
        
        # context embedding
        question_embed, context_embed = [], []
        for output, question_indices, context_indices in zip(
            outputs_embed, x["question_indices"], x["context_indices"]
        ):
            mean_question_embed = torch.mean(
                output[question_indices[0] : question_indices[1], :], dim=0
            ).unsqueeze(0)
            question_embed.append(
                mean_question_embed.repeat(context_indices[1] - context_indices[0], 1)
            )

            context_embed.append(output[context_indices[0] : context_indices[1], :])
        question_embed = pad_sequence(
            question_embed, batch_first=True
        )  # [batch_size, seq_len, text_output_layer_dim]
        context_embed = pad_sequence(
            context_embed, batch_first=True
        )  # [batch_size, seq_len, text_output_layer_dim]

        # span logits
        span_output, _ = self.span_attn(
            question_embed, context_embed, context_embed
        )  # [batch_size, seq_len, text_output_layer_dim]
        span_logits = self.span_output_layer(F.leaky_relu(span_output)).permute(
            0, 2, 1
        )  # [batch_size, 2, seq_len]
        span_outputs = F.softmax(span_logits, dim=1)

        return {
            "lm_loss": lm_loss,
            "span_logits": span_logits,
            "span_outputs": span_outputs,
            "span_preds": torch.argmax(span_outputs, dim=1, keepdim=False),
        }

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=self.config.gamma
        )

    def train_model(
        self,
        train_dataloader,
        val_dataloader,
        max_steps_per_train_epoch=float("inf"),
        max_steps_per_val_epoch=float("inf"),
        checkpoint_steps=None,
    ):
        self.configure_optimizers()

        train_loss, train_lm_loss, train_span_loss, train_span_acc = [], [], [], []
        val_loss, val_lm_loss, val_span_loss, val_span_acc = [], [], [], []
        epoch_times = []
        best_val_loss = float("inf")
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.perf_counter()
            self.train()
            epoch_loss, epoch_lm_loss, epoch_span_loss, epoch_span_acc = (
                0.0,
                0.0,
                0.0,
                0.0,
            )
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch + 1}",
                total=len(train_dataloader),
                dynamic_ncols=True,
            )
            for step, batch in enumerate(train_dataloader):
                # To control the max steps per epoch. This is used for debugging.
                if step >= max_steps_per_train_epoch:
                    break

                output_dict = self(batch)
                loss_dict = self.loss(output_dict, batch)

                epoch_loss += loss_dict["loss"].detach().float()
                epoch_lm_loss += loss_dict["lm_loss"].detach().float()
                epoch_span_loss += loss_dict["span_loss"].detach().float()
                epoch_span_acc += loss_dict["span_acc"].detach().float()

                loss_dict["loss"].backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                pbar.update(1)
                pbar.set_description(
                    (
                        f"Training Epoch: {epoch+1}/{self.config.num_epochs} "
                        + f"(loss: {loss_dict['loss'].detach().float():.3f}, "
                        + f"span_acc: {loss_dict['span_acc'].detach().float():.3f})"
                    )
                )
                
                if checkpoint_steps and step % checkpoint_steps == 0:
                    print(f"Saving checkpoint at step {step}...")
                    torch.save(
                        self.state_dict(), 
                        self.config.step_ckpt_file.format(epoch=epoch+1, step=step)
                    )
                    print(f"Model is saved in {self.config.step_ckpt_file.format(epoch=epoch+1, step=step)}.")
            
            pbar.close()
            epoch_times.append(time.perf_counter() - epoch_start_time)
            
            train_loss.append(epoch_loss / len(train_dataloader))
            train_lm_loss.append(epoch_lm_loss / len(train_dataloader))
            train_span_loss.append(epoch_span_loss / len(train_dataloader))
            train_span_acc.append(epoch_span_acc / len(train_dataloader))

            self.scheduler.step()

            print(
                (
                    f"Epoch {epoch + 1}: train_loss={train_loss[-1]:.4f}, "
                    + f"train_span_acc={train_span_acc[-1]:.4f}, "
                    + f"epoch time {epoch_times[-1]:.3f}s"
                )
            )

            # evaluate
            (
                val_epoch_loss,
                val_epoch_lm_loss,
                val_epoch_span_loss,
                val_epoch_span_acc,
            ) = self.evaluate_model(val_dataloader, max_steps_per_val_epoch)
            print(f"Current val loss: {val_epoch_loss}; Best val loss: {best_val_loss}")
            if val_epoch_loss < best_val_loss:
                print("Saving new checkpoint...")
                torch.save(self.state_dict(), self.config.ckpt_file)
                print(f"Model is saved in {self.config.ckpt_file}.")
                best_val_loss = val_epoch_loss
                
                results = {"epoch": epoch + 1}
                results["epoch_train_loss"] = float(train_loss[-1].detach().cpu().numpy())
                results["epoch_train_lm_loss"] = float(train_lm_loss[-1].detach().cpu().numpy())
                results["epoch_train_span_loss"] = float(train_span_loss[-1].detach().cpu().numpy())
                results["epoch_train_span_acc"] = float(train_span_acc[-1].detach().cpu().numpy())
                results["epoch_val_loss"] = float(val_epoch_loss.detach().cpu().numpy())
                results["epoch_val_lm_loss"] = float(val_epoch_lm_loss.detach().cpu().numpy())
                results["epoch_val_span_loss"] = float(val_epoch_span_loss.detach().cpu().numpy())
                results["epoch_val_span_acc"] = float(val_epoch_span_acc.detach().cpu().numpy())
                results["epoch_time"] = epoch_times[-1]
                json.dump(results, open(self.config.results_output_file, "w"), indent=4)
                

            val_loss.append(val_epoch_loss)
            val_lm_loss.append(val_epoch_lm_loss)
            val_span_loss.append(val_epoch_span_loss)
            val_span_acc.append(val_epoch_span_acc)

        avg_train_loss = sum(train_loss) / len(train_loss)
        avg_train_lm_loss = sum(train_lm_loss) / len(train_lm_loss)
        avg_train_span_loss = sum(train_span_loss) / len(train_span_loss)
        avg_train_span_acc = sum(train_span_acc) / len(train_span_acc)
        avg_val_loss = sum(val_loss) / len(val_loss)
        avg_val_lm_loss = sum(val_lm_loss) / len(val_lm_loss)
        avg_val_span_loss = sum(val_span_loss) / len(val_span_loss)
        avg_val_span_acc = sum(val_span_acc) / len(val_span_acc)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)

        results = {}
        results["avg_train_loss"] = float(avg_train_loss.detach().cpu().numpy())
        results["avg_train_lm_loss"] = float(avg_train_lm_loss.detach().cpu().numpy())
        results["avg_train_span_loss"] = float(avg_train_span_loss.detach().cpu().numpy())
        results["avg_train_span_acc"] = float(avg_train_span_acc.detach().cpu().numpy())
        results["avg_val_loss"] = float(avg_val_loss.detach().cpu().numpy())
        results["avg_val_lm_loss"] = float(avg_val_lm_loss.detach().cpu().numpy())
        results["avg_val_span_loss"] = float(avg_val_span_loss.detach().cpu().numpy())
        results["avg_val_span_acc"] = float(avg_val_span_acc.detach().cpu().numpy())
        results["avg_epoch_time"] = avg_epoch_time

        return results

    def evaluate_model(self, val_dataloader, max_steps_per_epoch=float("inf")):
        self.eval()
        epoch_loss, epoch_lm_loss, epoch_span_loss, epoch_span_acc = 0.0, 0.0, 0.0, 0.0
        for step, batch in enumerate(
            tqdm(
                val_dataloader,
                colour="green",
                desc=f"Evaluating Epoch",
                dynamic_ncols=True,
            )
        ):
            # To control the max steps per epoch. This is used for debugging.
            if step >= max_steps_per_epoch:
                break
            with torch.no_grad():
                output_dict = self(batch)
                loss_dict = self.loss(output_dict, batch)

                epoch_loss += loss_dict["loss"].detach().float()
                epoch_lm_loss += loss_dict["lm_loss"].detach().float()
                epoch_span_loss += loss_dict["span_loss"].detach().float()
                epoch_span_acc += loss_dict["span_acc"].detach().float()

        epoch_loss = epoch_loss / len(val_dataloader)
        epoch_lm_loss = epoch_lm_loss / len(val_dataloader)
        epoch_span_loss = epoch_span_loss / len(val_dataloader)
        epoch_span_acc = epoch_span_acc / len(val_dataloader)

        return epoch_loss, epoch_lm_loss, epoch_span_loss, epoch_span_acc

    def loss(self, output_dict, batch):
        # span loss
        span_loss = self.celoss(output_dict["span_logits"], batch["span_labels"])
        span_acc = self.binary_accuracy(output_dict["span_preds"], batch["span_labels"])
        # total loss
        total_loss = (output_dict["lm_loss"] + span_loss) / 2

        return {
            "loss": total_loss,
            "lm_loss": output_dict["lm_loss"],
            "span_loss": span_loss,
            "span_acc": span_acc,
        }