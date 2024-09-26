import os
import sys
sys.path.append("..")

from utils import configure_gpu_device, Base_Model_Dict

PROJECT_DIR = f"{os.path.dirname(__file__)}/.."


class Inference_Flan_T5_Config:
    batch_size = 1
    devices = [1]
    
    def __init__(self, dataset, batch, base_model, ckpt_dir=None):
        configure_gpu_device(self)
        
        self.batch = batch
        self.ckpt_dir = ckpt_dir
        
        self.base_model = base_model
        self.base_model_id = Base_Model_Dict[base_model]
        self.llama_based_model = False

        output_dir = f"{PROJECT_DIR}/inference/flan-t5/{dataset}_outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file_prefix = "finetuned_" if ckpt_dir else ""
        output_file_prefix = f"{base_model}_{output_file_prefix}"
        self.gold_output_file = f"{output_dir}/{output_file_prefix}{batch}.json"
        self.predictions_output_file = f"{output_dir}/{output_file_prefix}{batch}_predictions.json"
