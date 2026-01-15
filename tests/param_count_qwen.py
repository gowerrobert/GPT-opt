import os
import yaml
import torch
from transformers import Qwen2MoeConfig, AutoModelForCausalLM

def load_yaml(path):
    with open(path, "r") as f:
        y = yaml.safe_load(f)
    return y["config"]

def main():
    repo = os.path.dirname(os.path.dirname(__file__))
    cfg_path = os.path.join(repo, "hydra_conf", "model", "qwen2moe-small.yaml")
    cfg = load_yaml(cfg_path)
    conf = Qwen2MoeConfig(**cfg)
    m = AutoModelForCausalLM.from_config(conf)
    total = sum(p.numel() for p in m.parameters())
    print(f"Params: {total:,}")
    # Print parameter (layer) names and their shapes / counts
    for name, param in m.named_parameters():
        print(f"{name}: shape={tuple(param.shape)}, count={param.numel():,}")

if __name__ == "__main__":
    main()