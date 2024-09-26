import os
import sys
sys.path.append("..")

from utils import configure_gpu_device, Base_Model_Dict

PROJECT_DIR = f"{os.path.dirname(__file__)}/.."


class Inference_Config:
    batch_size = 1
    devices = [0]
    
    text_output_layer_dim = 512
    num_heads = 2
    
    def __init__(self, dataset, batch, base_model):
        configure_gpu_device(self)
        
        self.base_model = base_model
        self.base_model_id = Base_Model_Dict[base_model]
        self.llama_based_model = base_model in ["llama2", "alpaca"]
        
        self.batch = batch
        output_dir = f"{PROJECT_DIR}/inference/v2/{dataset}_outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.predictions_output_file = f"{output_dir}/{base_model}_{batch}_predictions.json"
        self.seq_tag_output_file = f"{output_dir}/{base_model}_{batch}_seqtag.json"