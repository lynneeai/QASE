# QASE

### This is the official repository of the EMNLP 2024 paper: [Enhancing Pre-Trained Generative Language Models with Question Attended Span Extraction on Machine Reading Comprehension](https://arxiv.org/pdf/2404.17991).


### Enviornment Setup
1. Use Python 3.9.
2. Run `pip install -r requirements.txt`.
3. Create a `.env` file, and put in `HF_TOKEN=YOUR_HUGGINGFACE_TOKEN`, and `OPENAI_API_KEY=YOUR_OPENAI_API_TOKEN`.
4. Get preprocessed data from [this driver folder](https://drive.google.com/file/d/1MG9HrAMOWvGAOyhVySs9iDAsBZO2NXo4/view?usp=sharing).


### Datasets
1. [MultiSpanQA](https://aclanthology.org/2022.naacl-main.90/): [leaderboard](https://multi-span.github.io/)
2. [SQuAD](https://arxiv.org/pdf/1606.05250): [leaderboard](https://rajpurkar.github.io/SQuAD-explorer/)
3. [Quoref](https://aclanthology.org/D19-1606/): [leaderboard](https://leaderboard.allenai.org/quoref/submissions/about)  

The preprocessed data of these datasets are available at [this driver folder](https://drive.google.com/file/d/1MG9HrAMOWvGAOyhVySs9iDAsBZO2NXo4/view?usp=sharing).


### Train and Fine-Tune

##### To train Flan-T5-Large<sub>QASE</sub> on, e.g. MultiSpanQA
1. Go to `configs/train_qase_config.py` and adjust any config settings if needed.
2. Go to the `train` directory.
3. Run `python train_quase.py --dataset multispanqa --base_model flan-t5-large`.
The trained model weights will be stored in the specified `ckpt_file` in the config settings. We release the best performing Flan-T5-Large<sub>QASE</sub> model, and the model weights are available [here](https://drive.google.com/drive/folders/1eYXAMaCbh_HR2zXjx8nevWtd3qeIgSIi?usp=sharing).

##### To fine-tune Flan-T5-Large on, e.g. MultiSpanQA
1. Go to `configs/finetune_flan_t5_config.py` and adjust any config settings if needed.
2. Go to the `train` directory.
3. Run `python finetune_flan_t5.py --dataset MultiSpanQA --base_model flan-t5-large`.
The trained LoRA weights will be stored in the specified `ckpt_dir` in the config settings.

##### To fine-tune Llama 2/Alpaca on, e.g. MultiSpanQA, with LoRA
1. Go to `configs/finetune_llama_config.py` and adjust any config settings if needed.
2. Go to the `train` directory.
3. Run `python finetune_llama.py --dataset MultiSpanQA --base_model llama2`.
The trained LoRA weights will be stored in the specified `ckpt_dir` in the config settings.


### Inference and Evaluate

##### To perform inference on the test/val batch of, e.g. MultiSpanQA, with a trained Flan-T5-Large<sub>QASE</sub> model weights
1. Go to `configs/inference_qase_config.py` and adjust any config settings if needed.
2. Go to the `inference/qase` directory.
3. Run `python inference_qase.py --dataset multispanqa --batch val --base_model flan-t5-large --ckpt_file {ckpt_file}`.
The output prediction file will be stored in the specified `predictions_output_file` in the config settings.

##### To perform inference on the test/val batch of, e.g. MultiSpanQA with zero-shot Flan-T5-Large or fine-tuned Flan-T5-Large
1. Go to `configs/inference_flan_t5_config.py` and adjust any config settings if needed.
2. Go to the `inference/flan-t5` directory.
3. To run with zero-shot, run `python inference_flan_t5.py --dataset multispanqa --batch test --base_model flan-t5-large`.
4. To run fine-tuned model, add `--ckpt_dir {ckpt_dir}`.
The output prediction file will be stored in the specified `predictions_output_file` in the config settings.

##### To perform inference on the test/val batch of, e.g. MultiSpanQA with zero-shot Llama 2 or fine-tuned Llama 2
1. Go to `configs/inference_llama_config.py` and adjust any config settings if needed.
2. Go to the `inference/llama` directory.
3. To run with zero-shot, run `python inference_llama.py --dataset multispanqa --batch test --base_model llama2`.
4. To run fine-tuned model, add `--ckpt_dir {ckpt_dir}`.
The output prediction file will be stored in the specified `predictions_output_file` in the config settings.

##### To run eval scripts
1. Go to the `inference` directory.
2. For MultiSpanQA, run `python multispanqa_eval_script.py --pred_file {pred_file} --gold_file ../datasets/MultiSpanQA/valid.json`.
3. For SQuAD, run `python squad_eval_script.py --pred_file {pred_file} --gold_file ../datasets/SQuAD/test_readable.json`.


### If you find our method useful, please cite us:
```
@article{ai2024enhancing,
  title={Enhancing Pre-Trained Generative Language Models with Question Attended Span Extraction on Machine Reading Comprehension},
  author={Ai, Lin and Hui, Zheng and Liu, Zizhou and Hirschberg, Julia},
  journal={arXiv preprint arXiv:2404.17991},
  year={2024}
}
```