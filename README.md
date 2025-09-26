# Partial Alignment as Reward-Guided Sampling (PARGS)

We will describe the detailed steps required to replicate the results for the summarization task in the "A Critical Look At Tokenwise
Reward-Guided Text Generation" paper.

## Overview:
1. Data preprocessing (Create partial sequences)
2. Train partial reward model
3. Train DPO reward model
4. Run baseline tests
5. Evaluation (Average Reward and GPT4 Evaluation)

## Setup
Install all the packages

```bash
cd PARGS
pip -r requirements.txt
```

## Step 1: Data Preprocessing & Create Partial Sequence Preference Dataset
Run the following command to process TL;DR dataset. Specifically we only keep the text in each post and create a dataset contains the "chosen" and "rejected" only. For partial sequences, we pad the shorter one of the "chosen" and "rejected" to the same length as the longer one, then take the partial (prefix) response of every length starting from 1, each pair of prefixes forms an entry in the partial dataset. 

```bash
python DataProcess/tldr_data_process.py
python DataProcess/tldr_data_process_partial.py
```

## Step 2: Train Partial Reward Model
Run the following command to train the partial reward model, based on DeBerta-v3-large reward model on the partial TL;DR dataset.
```bash
python partial_reward_training/tldr_partial_reward_model.py \
    --model_name_or_path="OpenAssistant/reward-model-deberta-v3-large" \
    --output_dir="partial_reward_modeling_tldr" \
    --per_device_train_batch_size=16 \
    --num_train_epochs=1 \
    --gradient_accumulation_steps=16 \
    --gradient_checkpointing=True \
    --learning_rate=5e-6 \
    --report_to="wandb" \
    --remove_unused_columns=False \
    --optim="adamw_torch" \
    --logging_steps=10 \
    --evaluation_strategy="steps" \
    --max_length=512 \
```

## Step 3: Train DPO Model
Run the following command to train the DPO model, based on the vistagi/gpt2-large-tldr-sum model on the processed (full) TL;DR dataset.

```bash
python DPO_training/GPT2_large_DPO_tldr.py \
    --model_name_or_path="vistagi/gpt2-large-tldr-sum" \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 16 \
    --logging_steps 10 \
    --output_dir="GPT2_large_tldr_DPO" \
    --report_to wandb \
    --fp16 \
    --logging_first_step \
    --no_remove_unused_columns
    --use_peft 
    --lora_r=16 
    --lora_alpha=16 
    --max_length=512 
    --max_prompt_length=512
    --optim="rmsprop"
```

**Note**: The script above may experience error with elder versions of the TRL library. Make sure to install the latest dev version by 
```bash
pip install git+https://github.com/huggingface/trl.git
```


## Step 4: Run Baseline Tests
Run the following command to get all the generation results and their reward scores based on the full-sequence reward models from all the baselines (ARGS, DPO, PPO, PARGS, PARGS-G, BESTN, TOP-K). Note we use the pretrained PPO model "vistagi/gpt2-large-tldr-sum-rlhf" for the PPO baseline. The generated sentences are stored in .json files while the reward scores are stored in one .csv file.
```bash
python PARGS_decode/PARGS_baselines.py
```

## Step 5: Evaluation
### Average Reward
Run the following command to get the table figures reporting the average reward and standard error.
```bash
python Evaluation/show_result.py
```

### GPT4-Evaluation
Run the following command to get the GPT-4 evaluation comparing two methods. For instance, to compare PARGS and ARGS, run
```bash
python Evaluation/GPT4_eval_tldr.py  "./generation_pargs.json"  "./generation_args.jso"
```




