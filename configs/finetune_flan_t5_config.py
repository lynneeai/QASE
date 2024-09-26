import os
import sys
sys.path.append("..")

from utils import configure_gpu_device, Base_Model_Dict

PROJECT_DIR = f"{os.path.dirname(__file__)}/.."


class Finetune_Flan_T5_Config:
    batch_size = 3
    num_epochs = 3
    lr = 1e-4
    weight_decay = 0.0
    gamma = 0.85
    devices = [1]
    experiment_name = None
        
    def __init__(self, dataset, base_model):
        configure_gpu_device(self)
        
        self.base_model = base_model
        self.base_model_id = Base_Model_Dict[base_model]
        self.llama_based_model = False
        
        output_file_suffix = f"_{self.experiment_name}" if self.experiment_name else ""
        self.output_dir = f"{PROJECT_DIR}/model_checkpoints/{base_model}/{dataset}"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.ckpt_dir = f"{self.output_dir}/checkpoint{output_file_suffix}"
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        self.results_output_file = f"{self.ckpt_dir}/results{output_file_suffix}.json"
