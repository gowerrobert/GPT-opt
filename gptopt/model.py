from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from .gpt_model import GPT as BaseGPT, GPTConfig as BaseGPTConfig
from .modded_gpt_model import GPT as ModGPT, GPTConfig as ModGPTConfig
import torch.nn as nn
import torch

class HuggingFaceCausalLMAdapter(nn.Module):
    """
    Adapter to match the (logits, loss) tuple API and (idx, labels=None, return_logits=True)
    signature expected by train.py and gpt_model.GPT.
    - Converts labels==-1 to -100 (HF ignore_index) before calling the HF model.
    - Returns (None, loss) if return_logits=False.
    - Proxies generate() to the underlying model.
    """
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, idx, labels=None, return_logits=True):
        # Translate ignore_index -1 -> -100 for HF loss
        labels_for_hf = None
        if labels is not None:
            if labels.dtype != torch.long:
                labels = labels.long()
            labels_for_hf = labels
            if (labels_for_hf == -1).any():
                labels_for_hf = labels_for_hf.clone()
                labels_for_hf[labels_for_hf == -1] = -100

        outputs = self.hf_model(input_ids=idx, labels=labels_for_hf) if labels is not None \
                  else self.hf_model(input_ids=idx)

        # HuggingFace returns ModelOutput with .logits and optional .loss
        logits = getattr(outputs, "logits", outputs[0] if isinstance(outputs, (tuple, list)) else None)
        loss = getattr(outputs, "loss", outputs[0] if labels is not None and isinstance(outputs, (tuple, list)) else None)

        # Inference-time optimization: only keep last token logits if no labels
        if labels is None and logits is not None:
            logits = logits[:, [-1], :]

        if not return_logits:
            logits = None
        return logits, loss

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.hf_model.generate(*args, **kwargs)

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
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_model_huggingface(config, device):
    # Random init only
    if 'model_name' in config['gpt_model']:
        hf = config['gpt_model']
        print(f"Random-initializing architecture for {hf['model_name']}")
        model_config = AutoConfig.from_pretrained(hf['model_name'], trust_remote_code=hf.get('trust_remote_code', False))
        model = AutoModelForCausalLM.from_config(model_config).to(device)
    else:
        g = config['gpt_model']
        model_config = GPT2Config(
            n_embd=g['n_embd'],
            n_layer=g['n_layer'],
            n_head=g['n_head'],
            vocab_size=g['vocab_size'],
        )
        model = GPT2LMHeadModel(model_config).to(device)
    return HuggingFaceCausalLMAdapter(model)

def load_model(model_name, config, device):
    if 'huggingface' in model_name:
        model = load_model_huggingface(config, device)
    elif 'modded-gpt' in model_name:
        gptconfig = ModGPTConfig(**config)
        model = ModGPT(gptconfig, device)
    else:
        gptconfig = BaseGPTConfig(**config)
        model = BaseGPT(gptconfig, device)
    return model

