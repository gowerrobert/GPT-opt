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
        model = HuggingFaceCausalLMAdapter(model)
    else:
        g = config['gpt_model']
        model_config = GPT2Config(
            n_embd=g['n_embd'],
            n_layer=g['n_layer'],
            n_head=g['n_head'],
            vocab_size=g['vocab_size'],
        )
        model = GPT2LMHeadModel(model_config).to(device)
        model = HuggingFaceCausalLMAdapter(model)  # Wrap for API consistency
        print("Loading gpt2 tokenizer as default tokenizer\n")
        tokenizer = AutoTokenizer.from_pretrained(g.get('gpt2', 'gpt2'))
    # Ensure tokenizer has a pad token and propagate to HF model config so attention_mask can be built
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if isinstance(model, HuggingFaceCausalLMAdapter) and getattr(model.hf_model.config, "pad_token_id", None) is None:
        model.hf_model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer

def load_model_huggingface(model_name, config, device):
    # If explicit dims are provided, build a Qwen2MoeConfig directly
    dim_keys = {'vocab_size','hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads'}
    
    if "Qwen2Moe" in model_name:
        if dim_keys.issubset(config.keys()):
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
            model_config = Qwen2MoeConfig(
                vocab_size=config['vocab_size'],
                hidden_size=config['hidden_size'],
                intermediate_size=config['intermediate_size'],
                num_hidden_layers=config['num_hidden_layers'],
                num_attention_heads=config['num_attention_heads'],
                num_key_value_heads=config['num_key_value_heads'],
            )
            return AutoModelForCausalLM.from_config(model_config).to(device)
    # Fallback: use a pretrained config name to construct architecture, random-init weights
    else:
        try:
            print(f"Random-initializing architecture for {model_name}")
            model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=hf.get('trust_remote_code', True))
            return AutoModelForCausalLM.from_config(model_config).to(device)

        except Exception as e:
            print(f"Failed to load model {model_name} from pretrained configs: {e}")

def load_model(model_name, config, device):
    # import pdb; pdb.set_trace()
    if ('source' in config) and (config['source'] == 'huggingface'):
        model = load_model_huggingface(model_name, config, device)
    elif 'modded-gpt' in model_name:
        gptconfig = ModGPTConfig(**config)
        model = ModGPT(gptconfig, device)
    else:
        gptconfig = BaseGPTConfig(**config)
        model = BaseGPT(gptconfig, device)
    return model

