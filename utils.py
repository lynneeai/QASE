import os
from dotenv import load_dotenv
from huggingface_hub import login


Base_Model_Dict = {
    "llama2": "meta-llama/Llama-2-7b-hf",
    "alpaca": "decapoda-research/llama-7b-hf",
    "flan-t5-small": "google/flan-t5-small",
    "flan-t5-base": "google/flan-t5-base",
    "flan-t5-large": "google/flan-t5-large",
    "flan-t5-xl": "google/flan-t5-xl",
    "flan-t5-xxl": "google/flan-t5-xxl"
}


def find_all_substrings(s, sub):
    indices = []
    start = 0
    while True:
        start = s.find(sub, start)
        if start == -1:
            break
        indices.append((start, start + len(sub)))
        start += len(sub)
    return indices


def find_all_sublists(l, sub):
    indices = []
    for start in range(len(l)):
        if l[start : start + len(sub)] == sub:
            indices.append((start, start + len(sub)))
    return indices


def longest_overlapping_substring(str1, str2):
    n = len(str1)
    m = len(str2)

    # Initialize a table to store results of sub-problems
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # To store the length of the longest overlapping substring
    max_len = 0

    # To store the ending index of the longest overlapping substring in str1
    end_index = 0

    # Fill the dp table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1

                # Update the length and ending index of the maximum length of overlapping substring
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_index = i
            else:
                dp[i][j] = 0

    # If there is no overlapping substring
    if max_len == 0:
        return ""

    # Return the longest overlapping substring
    return str1[end_index - max_len : end_index]


def set_llama2_credential():
    load_dotenv()
    login(os.getenv("HF_TOKEN"))


def configure_gpu_device(config):
    """
    Set cuda visible deviceds and re-index config.devices
    This is required for device_map="auto" model loading,
    because cuda tries to get all available GPUs available,
    so CUDA_VISIBLE_DEVICES should be set to specified devices.
    After setting CUDA_VISIBLE_DEVICES, torch device index will be reset to start from 0.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in config.devices)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config.devices = [i for i in range(len(config.devices))]
