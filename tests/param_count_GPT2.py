import os
import torch

from gptopt.gpt_model import GPT, GPTConfig

def load_yaml_config(path):
    try:
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return data.get("config", {})
    except Exception as e:
        print(f"Warning: failed to read YAML at {path}: {e}")
        return {}

def main():
    # Default to repo-relative path
    repo_root = os.path.dirname(os.path.dirname(__file__))
    yaml_path = os.path.join(repo_root, "hydra_conf", "model", "gpt-small.yaml")

    cfg = {
        "block_size": 1024,
        "vocab_size": 50257,
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "no_layernorm": False,
        "flash_attention": True,
    }
    cfg.update(load_yaml_config(yaml_path))

    device = torch.device("cpu")
    model = GPT(GPTConfig(**cfg), device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,}")

if __name__ == "__main__":
    main()