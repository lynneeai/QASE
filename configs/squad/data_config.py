import sys
sys.path.append("../..")

from utils import Base_Model_Dict


class SQuAD_Data_Config:
    def __init__(self, batch_size, llama_based_model=True, base_model="llama2"):
        self.batch_size = batch_size
        self.llama_based_model = llama_based_model
        
        self.base_model = base_model
        self.base_model_id = Base_Model_Dict[base_model]