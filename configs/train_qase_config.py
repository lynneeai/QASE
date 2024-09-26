import os
import sys
sys.path.append("..")

from utils import configure_gpu_device, Base_Model_Dict

PROJECT_DIR = f"{os.path.dirname(__file__)}/.."


class Train_Config:
    batch_size = 3
    num_epochs = 3
    lr = 1e-4
    weight_decay = 0.0
    gamma = 0.85
    devices = [0]

    text_output_layer_dim = 512
    num_heads = 2
    
    experiment_name = None
    
    def __init__(self, dataset, base_model, save_checkpoint_after_steps=False):
        configure_gpu_device(self)
        
        self.base_model = base_model
        self.base_model_id = Base_Model_Dict[base_model]

        output_dir = f"{PROJECT_DIR}/model_checkpoints/{base_model}_spanqa_v2/{dataset}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file_suffix = f"_{self.experiment_name}" if self.experiment_name else ""
        self.ckpt_file = f"{output_dir}/model{output_file_suffix}.pth"
        self.results_output_file = f"{output_dir}/results{output_file_suffix}.json"
        
        if save_checkpoint_after_steps:
            step_ckpt_output_dir = f"{output_dir}/step_checkpoint"
            if not os.path.exists(step_ckpt_output_dir):
                os.makedirs(step_ckpt_output_dir)
            self.step_ckpt_file = f"{step_ckpt_output_dir}/" + "epoch_{epoch}_step_{step}_ckpt_model" + f"{output_file_suffix}.pth"
        
        self.llama_based_model = base_model in ["llama2", "alpaca"]