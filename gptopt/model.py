from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from .gpt_model import GPT as BaseGPT, GPTConfig as BaseGPTConfig
from .modded_gpt_model import GPT as ModGPT, GPTConfig as ModGPTConfig
import torch.nn as nn
import torch

def load_model_and_tokenizer(config, device):
    # Always random init weights
    if 'model_name' in config['gpt_model']:
        hf = config['gpt_model']
        print(f"Random-initializing architecture for {hf['model_name']}")
        model_config = AutoConfig.from_pretrained(hf['model_name'], trust_remote_code=hf.get('trust_remote_code', False))
        model = AutoModelForCausalLM.from_config(model_config).to(device)
        tokenizer = AutoTokenizer.from_pretrained(hf['model_name'], trust_remote_code=hf.get('trust_remote_code', False))
    else:
        g = config['gpt_model']
        model_config = GPT2Config(
            n_embd=g['n_embd'],
            n_layer=g['n_layer'],
            n_head=g['n_head'],
            vocab_size=g['vocab_size'],
        )
        model = GPT2LMHeadModel(model_config).to(device) 
        print("Loading gpt2 tokenizer as default tokenizer\n")
        tokenizer = AutoTokenizer.from_pretrained(g.get('gpt2', 'gpt2'))
    # Ensure tokenizer has a pad token and propagate to HF model config so attention_mask can be built
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token 
    return model, tokenizer

def load_model_huggingface(model_name, config, device):
    # Simplified: build Qwen2Moe directly from YAML fields, like tests/param_count_qwen.py
    if "Qwen2Moe" in model_name:
        try:
            from transformers import Qwen2MoeConfig
        except Exception as e:
            raise ImportError("Qwen2MoeConfig not available. Please upgrade transformers.") from e
        
            # Basic sanity checks
            hs = config['hidden_size']
            if hs % config['num_attention_heads'] != 0:
                raise ValueError("hidden_size must be divisible by num_attention_heads")
            if hs % config['num_key_value_heads'] != 0:
                raise ValueError("hidden_size must be divisible by num_key_value_heads")

            print(f"Random-initializing architecture for Qwen2Moe (custom config from Hydra: {model_name})")
        model_config = Qwen2MoeConfig(**config)
        return AutoModelForCausalLM.from_config(model_config).to(device)

    # Fallback: use a pretrained config id (random init)
    try:
        trust = config.get('trust_remote_code', True) if isinstance(config, dict) else True
        print(f"Random-initializing architecture for {model_name}")
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust)
        return AutoModelForCausalLM.from_config(model_config).to(device)
    except Exception as e:
        print(f"Failed to load model {model_name} from pretrained configs: {e}")
        raise

def load_model(model_name, config, device):

    if ('source' in config) and (config['source'] == 'huggingface'):
        model = load_model_huggingface(model_name, config, device)
    elif 'modded-gpt' in model_name:
        gptconfig = ModGPTConfig(**config)
        model = ModGPT(gptconfig, device)
    else:
        gptconfig = BaseGPTConfig(**config)
        model = BaseGPT(gptconfig, device)
    return model

