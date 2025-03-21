import yaml
import argparse
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel
import importlib
from matsign.svd import matsign as svd_matsign  # Import the SVD-based matsign function

def apply_function_to_weights(model, func, results, steps = 5):
    """
    Apply a numerical linear algebra function to the weights of every linear layer in the model
    and record the difference with the SVD-based matsign function.

    Parameters
    ----------
    model : torch.nn.Module
        The GPT-2 model.
    func : callable
        The numerical linear algebra function to apply.
    results : dict
        Dictionary to store the differences for each layer.
    """
    for name, param in model.named_parameters():
        if "weight" in name and param.ndim == 2:  # Apply only to 2D weight matrices
            print(f"Processing {name}")
            original_weights = param.data.clone()
            svd_result = svd_matsign(original_weights)  # Compute SVD-based matsign
            func_result = func(original_weights, steps =steps)  # Compute matsign using the chosen function
            relative_difference = (svd_result - func_result).norm().item()  # Compute norm of the difference
            results[name] = relative_difference/svd_result.norm().item()
            print(f"Relative Diff for {name}: {relative_difference}")

def main(config_file):
    # Load configuration
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if 'model_name' in config['gpt_model']:
        model = AutoModelForCausalLM.from_pretrained(config['gpt_model']['model_name'], device_map="auto").to(device)
    else:
        gpt_config = config['gpt_model']
        model_config = GPT2Config(
            n_embd=gpt_config['n_embd'],
            n_layer=gpt_config['n_layer'],
            n_head=gpt_config['n_head'],
            vocab_size=gpt_config['vocab_size'],
        )
        model = GPT2LMHeadModel(model_config).to(device)

    # Print model details
    num_layers = len([name for name, _ in model.named_parameters() if "weight" in name and "h." in name])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of layers: {num_layers}")
    print(f"Total number of parameters: {total_params}")

    # Print number of linear layers
    num_linear_layers = len([name for name, param in model.named_parameters() if "weight" in name and param.ndim == 2])
    print(f"Number of linear layers: {num_linear_layers}")

    # Dynamically import the specified matsign function
    matsign_method = config['matsign_method']['name']
    steps = config['matsign_method']['steps']
    module_name = f"matsign.{matsign_method}"
    module = importlib.import_module(module_name)
    func = getattr(module, "matsign")

    # Record differences
    results = {}
    apply_function_to_weights(model, func, results, steps = steps)
    # Save results to file
    name = config['name'] +  ".json"
    name = "matsign/outputs/" + name
    with open( name, "w") as f: 
        json.dump(results, f, indent=4)
    print(f"Results saved to {name}")

if __name__ == "__main__":
    # Argument parser to provide config file
    parser = argparse.ArgumentParser(description='Test numerical linear algebra functions on GPT-2 weights.')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    
    args = parser.parse_args()
    main(args.config)
