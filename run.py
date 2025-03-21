import yaml
import argparse
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel
import importlib
from matsign.svd import matsign as svd_matsign  # Import the SVD-based matsign function
from matsign.data import load_data  # Import the data loading function
import time  # Import time module for measuring execution time

def apply_function_to_gradients(model, func, dataloader, steps=5):
    """
    Apply a numerical linear algebra function to the gradients of every linear layer in the model
    and record the average relative error, average time taken, and average spectral norm error.

    Parameters
    ----------
    model : torch.nn.Module
        The GPT-2 model.
    func : callable
        The numerical linear algebra function to apply.
    dataloader : DataLoader
        The DataLoader for the dataset.
    steps : int
        Number of steps for the numerical method.

    Returns
    -------
    dict
        Dictionary containing the average relative error, average time taken, and average spectral norm error
        for each layer.
    """
    results = {name: {"avg_relative_error": 0.0, "avg_time": 0.0, "avg_spectral_error": 0.0, "count": 0}
               for name, param in model.named_parameters() if "weight" in name and param.ndim >= 2}

    model.train()
    for batch in dataloader:
        inputs = batch['text']  # Assuming the dataset provides text inputs
        inputs = inputs.to(next(model.parameters()).device)
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss
        loss.backward()  # Compute gradients

        for name, param in model.named_parameters():
            if "weight" in name and param.grad is not None and param.grad.ndim >= 2:  # Apply only to 2D gradients
                original_grad = param.grad.clone()
                svd_result = svd_matsign(original_grad)  # Compute SVD-based matsign

                start_time = time.time()
                func_result = func(original_grad, steps=steps)  # Compute matsign using the chosen function
                elapsed_time = time.time() - start_time

                relative_error = (svd_result - func_result).norm().item() / svd_result.norm().item()
                spectral_error = torch.linalg.norm(svd_result - func_result, ord=2).item()

                # Incrementally update results using the provided formula
                t = results[name]["count"]
                results[name]["avg_relative_error"] = (t / (1 + t)) * results[name]["avg_relative_error"] + (1 / (1 + t)) * relative_error
                results[name]["avg_time"] = (t / (1 + t)) * results[name]["avg_time"] + (1 / (1 + t)) * elapsed_time
                results[name]["avg_spectral_error"] = (t / (1 + t)) * results[name]["avg_spectral_error"] + (1 / (1 + t)) * spectral_error
                results[name]["count"] += 1

        model.zero_grad()  # Reset gradients after processing

    # Remove count from the final results
    for name in results:
        del results[name]["count"]

    return results

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

    # Load training parameters
    training_params = config['training_params']
    batch_size = training_params['batch_size']
    dataname = training_params['dataname']

    # Load dataset
    train_dataloader, _ = load_data(dataname, batch_size)

    # Dynamically import the specified matsign function
    matsign_method = config['matsign_method']['name']
    steps = config['matsign_method']['steps']
    module_name = f"matsign.{matsign_method}"
    module = importlib.import_module(module_name)
    func = getattr(module, "matsign")

    # Record differences
    results = apply_function_to_gradients(model, func, train_dataloader, steps=steps)
    # Save results to file
    name = config['name'] +  ".json"
    name = "matsign/outputs/" + name
    with open( name, "w") as f: 
        json.dump(results, f, indent=4)
    print(f"Results saved to {name}")

if __name__ == "__main__":
    # Argument parser to provide config file
    parser = argparse.ArgumentParser(description='Test numerical linear algebra functions on GPT-2 gradients.')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    
    args = parser.parse_args()
    main(args.config)
