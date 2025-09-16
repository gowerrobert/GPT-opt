import json
import re
import itertools
import fire
import os
from typing import List, Tuple, Dict, Optional

def parse_bash_script(bash_script: str) -> List[Tuple[str, str]]:
    """
    Parse the bash script to extract parameter names, default values, and their positions.

    Parameters:
        bash_script (str): Path to the bash script to parse.

    Returns:
        List[Tuple[str, str]]: A list of tuples where each tuple contains the parameter name and its default value, 
                               sorted by their positional order in the bash script.
    """
    params_with_defaults: Dict[int, Tuple[str, str]] = {}
    
    with open(bash_script, 'r') as file:
        for line in file:
            # Extract parameter assignment patterns like: param_name=${1:-default_value}
            match = re.search(r'(\w+)=(\$\{(\d+):-([^}]+)\})', line)
            if match:
                param_name = match.group(1)  # The parameter name (e.g., lr)
                position = int(match.group(3))  # The positional index (e.g., 1 for ${1})
                default_value = match.group(4)  # The default value (e.g., 5e-5)
                params_with_defaults[position] = (param_name, default_value)
    
    # Ensure parameters are sorted by their position
    sorted_params = [params_with_defaults[pos] for pos in sorted(params_with_defaults)]
    return sorted_params

def validate_params(expected_params: List[Tuple[str, str]], provided_params: List[str]) -> None:
    """
    Validate provided parameters from the JSON file against the expected parameters in the bash script.

    Parameters:
        expected_params (List[Tuple[str, str]]): List of tuples representing the expected parameter names and default values.
        provided_params (List[str]): List of parameter names provided in the JSON file.

    Raises:
        ValueError: If the JSON contains unexpected parameters that are not present in the bash script.
    """
    expected_param_names = {param[0] for param in expected_params}  # Extract expected parameter names
    unexpected_params = set(provided_params) - expected_param_names  # Find any unexpected params
    if unexpected_params:
        raise ValueError(f"Unexpected parameters in JSON: {unexpected_params}")

def generate_commands(bash_script: str, param_file: str, output_file: str, full_tasks: bool = True, add_logs: bool = True, log_dir: Optional[str] = None) -> None:
    """
    Generate a list of commands from the bash script and a JSON file containing custom parameter values.

    Parameters:
        bash_script (str): Path to the bash script.
        param_file (str): Path to the JSON file containing custom parameters and their values.
        output_file (str): Path to the output file where generated commands will be written.
        full_tasks (bool): If True, prepends the bash script to the generated command lines. Defaults to True.
        add_logs (bool): If True, appends numbered suffixes for stdout and stderr log files. Defaults to True.
        log_dir (Optional[str]): Directory where log files will be saved if add_logs is True. Defaults to None.

    Raises:
        ValueError: If there are validation errors between the bash script parameters and JSON parameters.
    """
    # Parse the bash script for parameter names, default values, and their positions
    sorted_params = parse_bash_script(bash_script)

    # Load custom parameters from the JSON file
    with open(param_file, 'r') as file:
        custom_params = json.load(file)

    # Validate the parameters from the JSON file against the ones expected from the bash script
    validate_params(sorted_params, custom_params.keys())

    # Merge default values with custom values from the JSON file
    values: List[List[str]] = []
    for param_name, default_value in sorted_params:
        if param_name in custom_params:
            values.append(custom_params[param_name])
        else:
            values.append([default_value])  # Use default if no custom value provided

    # Generate all combinations of parameter values
    combinations = list(itertools.product(*values))
    total_combinations = len(combinations)
    num_digits = len(str(total_combinations))  # Determine padding for numbers

    # Ensure log directory exists if specified and add_logs is True
    if add_logs and log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Write the generated command lines to the output file
    with open(output_file, 'w') as f:
        for i, combination in enumerate(combinations):
            # Generate the base command string with parameters
            args = " ".join(map(str, combination))
            command = f"./{bash_script} {args}" if full_tasks else args

            if add_logs:
                # Zero-padded file numbering for logs
                padded_index = str(i).zfill(num_digits)
                if log_dir:
                    command += f" > {log_dir}/log_{padded_index}.out 2> {log_dir}/log_{padded_index}.err"
                else:
                    command += f" > log_{padded_index}.out 2> log_{padded_index}.err"

            f.write(command + "\n")

    print(f"Commands written to {output_file}, number of lines: {total_combinations}.")

def count_lines_in_file(file_path: str) -> int:
    """
    Utility function to count the number of lines in a file.

    Parameters:
        file_path (str): Path to the file.

    Returns:
        int: The number of lines in the file.
    """
    with open(file_path, 'r') as file:
        return sum(1 for _ in file)

# if __name__ == '__main__':
#     params_path = 'configs/params.json'
#     bash_script_path = 'scripts/run_roberta-preln.sh'
#     output_file_path = 'configs/commands.txt'
#     generate_commands(bash_script_path, params_path, output_file_path)
#     num_lines = count_lines_in_file(output_file_path)
#     print(f"Number of command lines written: {num_lines}")

if __name__ == '__main__':
    fire.Fire(generate_commands)
