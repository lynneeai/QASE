import sys
sys.path.append("../..")
import os

from utils import Base_Model_Dict

PROJECT_DIR = f"{os.path.dirname(__file__)}/../.."


class MultiSpanQA_Data_Config:
    train_file = f"{PROJECT_DIR}/datasets/MultiSpanQA/train.json"
    val_file = f"{PROJECT_DIR}/datasets/MultiSpanQA/valid.json"
    test_file = f"{PROJECT_DIR}/datasets/MultiSpanQA/test.json"

    def __init__(self, batch_size, llama_based_model=True, base_model="llama2"):
        self.batch_size = batch_size
        self.llama_based_model = llama_based_model
        
        self.base_model = base_model
        self.base_model_id = Base_Model_Dict[base_model]
