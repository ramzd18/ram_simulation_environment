import atexit
import json
import math
import os
import random
import shutil
import string
import subprocess
import time
from typing import List, Optional, Tuple

import numpy as np
import requests
import torch
import torch.nn.functional as F
import wandb
from pydantic import BaseModel, Field
# from tenacity import retry, stop_after_attempt, wait_exponential
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

# Global variable to track vLLM process
vllm_process = None

def cleanup_vllm():
    global vllm_process
    if vllm_process:
        print("\nTerminating vLLM process...")
        vllm_process.terminate()
        try:
            vllm_process.wait(timeout=5)
            print("vLLM process terminated.")
        except subprocess.TimeoutExpired:
            print("vLLM process did not terminate gracefully, killing.")
            vllm_process.kill()
            vllm_process.wait()
            print("vLLM process killed.")
        vllm_process = None

atexit.register(cleanup_vllm)

class TrainingConfig(BaseModel):
    """Training configuration parameters"""
    model_name: str = Field(..., description="Base model to train")
    lr: float = Field(1e-5, description="Learning rate")
    training_steps: int = Field(10, description="Number of training steps")
    batch_size: int = Field(2, description="Batch size")
    group_size: int = Field(4, description="Group size")
    seq_len: int = Field(2048, description="Sequence length")
    gradient_accumulation_steps: int = Field(32, description="Gradient accumulation steps")
    device: str = Field("cuda" if torch.cuda.is_available() else "cpu", description="Training device")
    save_path: str = Field("trained_model_checkpoints", description="Checkpoint save path")
    vllm_restart_interval: int = Field(3, description="vLLM restart interval")
    vllm_port: int = Field(8001, description="vLLM server port")
    api_port: int = Field(5000, description="API server port")
    use_wandb: bool = Field(False, description="Enable wandb logging")
    wandb_project: Optional[str] = Field(None, description="Wandb project name")
    wandb_group: Optional[str] = Field(None, description="Wandb group name")

def setup_training(config: TrainingConfig):
    """Setup training environment"""
    response = requests.post(
        f"http://localhost:{config.api_port}/setup",
        timeout=10
    )
    if response.status_code != 200:
        raise Exception(f"Failed to setup training environment: {response.text}")

def start_collection(config: TrainingConfig):
    """Start data collection"""
    response = requests.post(
        f"http://localhost:{config.api_port}/start_collection",
        timeout=10
    )
    if response.status_code != 200:
        raise Exception(f"Failed to start collection: {response.text}")

def get_batch(config: TrainingConfig):
    """Get training batch from API"""
    response = requests.post(
        f"http://localhost:{config.api_port}/dispatch",
        timeout=10
    )
    if response.status_code != 200:
        return None
    return response.json()["data"]

def process_conversation_data(conversation_data):
    # Fix this later 
    tokens, masks, scores = [], [], []
    
    for conv, rewards, char1, char2, scenario in conversation_data:
        conversation_tokens = []
        conversation_masks = []
        conversation_scores = []
        
        for message in conv:
            # Tokenize message and create masks
            # Add to conversation_tokens, conversation_masks
            pass
            
        # Add rewards to scores
        conversation_scores = [rewards.get("score", 0.0) for _ in conv]
        
        tokens.extend(conversation_tokens)
        masks.extend(conversation_masks)
        scores.extend(conversation_scores)
    
    return {
        "tokens": tokens,
        "masks": masks,
        "scores": scores,
        "overrides": None  # Add if needed
    }


def pad_data_to_good_offset(data, batch_size: int):
    """Pad and prepare data for training"""
    max_token_len = max([max([len(x) for x in item["tokens"]]) for item in data["batch"]])
    good_multiple = 64
    if (max_token_len - 1) % good_multiple != 0:
        max_token_len = math.ceil((max_token_len - 1) / good_multiple) * good_multiple
        token_setup_len = max_token_len + 1
    else:
        token_setup_len = max_token_len
        max_token_len = max_token_len - 1

    input_ids, labels, advantages = [], [], []
    lengths = []
    
    for item in data["batch"]:
        scores = np.array(item["scores"])
        if len(scores) > 1:
            scores = scores - scores.mean()
            scores = scores / max(scores.std(), 1e-8)
        item["scores"] = scores
        
        if item["overrides"] is not None:
            for i in range(len(item["overrides"])):
                if item["overrides"][i].get("set_advantage_to_zero", False):
                    item["scores"][i] = 0

        for i in range(len(item["tokens"])):
            lengths.append(math.ceil((len(item["tokens"][i]) - 1) / good_multiple) * good_multiple)
            label_item = np.concatenate([
                np.array(item["masks"][i]),
                np.full(max(0, token_setup_len - len(item["tokens"][i])), -100, dtype=np.int32)
            ])
            item["tokens"][i] = np.concatenate([
                np.array(item["tokens"][i]),
                np.zeros(max(0, token_setup_len - len(item["tokens"][i])), dtype=np.int32)
            ])
            input_ids.append(item["tokens"][i][:-1])
            labels.append(label_item[1:])
            advantages.append(item["scores"][i])

    token_batches, label_batches, advantage_batches = [], [], []
    for i in range(len(input_ids) // batch_size):
        token_batches.append(torch.tensor(np.stack(input_ids[i * batch_size:(i + 1) * batch_size], axis=0)))
        label_batches.append(torch.tensor(np.stack(labels[i * batch_size:(i + 1) * batch_size], axis=0)))
        advantage_batches.append(torch.tensor(np.stack(advantages[i * batch_size:(i + 1) * batch_size], axis=0)).view(-1, 1))

    return token_batches, label_batches, advantage_batches

def register_trainer(config: TrainingConfig):
    """Register trainer with API"""
    requests.post(
        "http://localhost:8000/register",
        json={
            "wandb_group": config.wandb_group,
            "wandb_project": config.wandb_project,
            "batch_size": config.batch_size * config.gradient_accumulation_steps,
            "max_token_len": config.seq_len,
            "starting_step": 0,
            "checkpoint_dir": config.save_path,
            "save_checkpoint_interval": config.training_steps,
            "num_steps": config.training_steps,
        },
        timeout=10,
    )

# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
# def get_batch():
#     """Get training batch from API"""
#     return requests.get("http://localhost:8000/batch", timeout=10).json()

def load_initial_model(config: TrainingConfig):
    """Load initial model"""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16)
    model.to(config.device)
    model.gradient_checkpointing_enable()
    model.train()
    return model, tokenizer


def train(config: TrainingConfig):
    """Main training loop"""
    global vllm_process

    setup_training(config)
    start_collection(config)

    model, tokenizer = load_initial_model(config)

    optimizer = AdamW(model.parameters(), lr=config.lr)

    print(f"Starting training for {config.training_steps} steps on {config.device}")
    print(f"vLLM restart interval: {config.vllm_restart_interval} steps")

    os.makedirs(config.save_path, exist_ok=True)
    register_trainer(config)

    vllm_command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", config.model_name,
        "--port", str(config.vllm_port),
        "--dtype", "auto",
        "--gpu-memory-utilization", "0.45",
        "--disable-log-requests",
    ]
    
    try:
        vllm_process = subprocess.Popen(vllm_command)
        print(f"vLLM server launched with PID: {vllm_process.pid}")
    except Exception as e:
        print(f"Error launching vLLM: {e}")
        config.vllm_restart_interval = config.training_steps + 1
    batches = []
    while True: 
        for step in range(config.training_steps):
            have_data= False
            while not have_data:
                data= get_batch(config)
                if data is None:
                    time.sleep(15)
                else:
                    have_data= True
                    batches = data
                    if step <= config.training_steps -1: 
                        start_collection(config)

            print(f"Step {step+1}/{config.training_steps}")
            
            
            # for tokens, labels, advantages in zip(token_batches, label_batches, advantage_batches):
            #     tokens = tokens.to(config.device)
            #     labels = labels.to(config.device)
            #     advantages = advantages.to(config.device)

            #     outputs = model(tokens)
            #     logits = outputs.logits

            #     logp_per_token = -F.cross_entropy(
            #         logits.view(-1, logits.size(-1)),
            #         labels.view(-1),
            #         reduction="none",
            #         ignore_index=-100,
            #     ).view(labels.shape)

            #     mask = (labels != -100).float()
            #     grpo_loss_term = torch.exp(logp_per_token - logp_per_token.detach())
            #     grpo_loss = (
            #         ((-grpo_loss_term * mask).sum(-1) / mask.sum(-1))
            #         * advantages.to(logp_per_token.device)
            #     ).mean() / config.gradient_accumulation_steps

            #     grpo_loss.backward()
            #     total_loss += grpo_loss.item()

            # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # optimizer.step()
            # optimizer.zero_grad()

            if config.use_wandb:
                wandb.log({
                    "train/loss": total_loss,
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/grad_norm": grad_norm.item(),
                }, step=step + 1)

            print(f"Step Loss: {total_loss:.4f}")

            # Checkpoint and vLLM restart logic
            if (step + 1) % config.vllm_restart_interval == 0 or step == config.training_steps - 1:
                # Notify API about model reload
                response = requests.post(
                    f"http://localhost:{config.api_port}/reload",
                    timeout=10
                )
                if response.status_code != 200:
                    print(f"Warning: Failed to notify API about model reload: {response.text}")
                
                checkpoint_path = os.path.join(config.save_path, f"step_{step+1}")
                if os.path.exists(checkpoint_path):
                    shutil.rmtree(checkpoint_path)
                os.makedirs(checkpoint_path, exist_ok=True)
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)

                if vllm_process:
                    vllm_process.terminate()
                    try:
                        vllm_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        vllm_process.kill()
                        vllm_process.wait()
                    vllm_process = None

                vllm_command = [
                    "python", "-m", "vllm.entrypoints.openai.api_server",
                    "--model", checkpoint_path,
                    "--port", str(config.vllm_port),
                    "--dtype", "auto",
                    "--gpu-memory-utilization", "0.45",
                    "--disable-log-requests",
                    "--served-model-name", config.model_name,
                ]

                try:
                    vllm_process = subprocess.Popen(vllm_command)
                    print(f"vLLM server restarted with PID: {vllm_process.pid}")
                except Exception as e:
                    print(f"Error restarting vLLM: {e}")
                    config.vllm_restart_interval = config.training_steps + 1

    if config.use_wandb:
        wandb.finish()

    final_save_path = os.path.join(config.save_path, "final_model")
    if os.path.exists(final_save_path):
        shutil.rmtree(final_save_path)
    os.makedirs(final_save_path, exist_ok=True)
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print("Training completed and final model saved.")

if __name__ == "__main__":
    training_config = TrainingConfig(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        training_steps=20,
        vllm_restart_interval=3,
        api_port=5000,  # Add API port
        use_wandb=True,
        wandb_project="grpo-trainer-example",
    )
    train(training_config)
