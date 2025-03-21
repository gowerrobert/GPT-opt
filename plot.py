import json
import argparse
import matplotlib.pyplot as plt
import os
import yaml

def plot_matsign_differences(config_file):
    """
    Plot the differences between the matrix sign computed using two methods.

    Parameters
    ----------
    input_file : str
        Path to the JSON file containing the differences.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    # Load results
    name = config['name'] +  ".json"
    name = "matsign/outputs/" + name
    with open( name, "r") as f: 
        results = json.load(f)

    # Prepare data for plotting
    layers = [name.replace("transformer.", "").replace(".weight", "").strip() for name in results.keys()]
    differences = list(results.values())

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(layers, differences, color="skyblue")
    plt.xticks(rotation=90, fontsize=8)
    plt.xlabel("Layer")
    plt.ylabel("Norm Difference")
    plt.title("Matrix Sign Differences Across Layers")
    plt.tight_layout()

    # Save plot as PDF in the figures folder
    os.makedirs("figures", exist_ok=True)
    output_file = os.path.join("figures", config['name'] + ".pdf")
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    # Argument parser to provide input file
    parser = argparse.ArgumentParser(description='Plot matrix sign differences.')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    
    args = parser.parse_args()
    plot_matsign_differences(args.config)
