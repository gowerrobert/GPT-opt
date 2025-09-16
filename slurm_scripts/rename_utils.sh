#!/bin/bash

# Function to get the next available numbered name in a directory.
# Arguments:
#   $1 - Directory path where the file or directory will be located.
#   $2 - Base name for the new file or directory.
#   $3 - Extension for files (e.g., ".txt" or ".json"), or an empty string for directories.
# Returns:
#   The next available filename or directory name in the format "base_name_N.ext" where N is an incremented number.
get_next_name() {
    local base_dir="$1"
    local base_name="$2"
    local ext="$3"
    local counter=1

    # Increment the counter until a unique name is found
    while [[ -e "${base_dir}/${base_name}_${counter}${ext}" ]]; do
        ((counter++))
    done
    echo "${base_dir}/${base_name}_${counter}${ext}"
}

# Function to rename a file by incrementing its name to the next available number.
# Arguments:
#   $1 - Directory path where the original file is located.
#   $2 - Filename of the original file (e.g., "params.json").
#   $3 - Base name to use for the new file (without extension).
# Behavior:
#   Parses the file extension from the original filename and renames it to the next available name
#   using "base_name_N.ext" format, where N is an incremented number.
rename_file() {
    local base_dir="$1"
    local filename="$2"
    local base_name="$3"

    # Full path to the original file
    local original_file="${base_dir}/${filename}"

    # Check if the original file exists
    if [[ ! -e "$original_file" ]]; then
        echo "Warning: $original_file does not exist. Skipping renaming."
        return 1
    fi

    # Parse the file extension
    local ext="${filename##*.}"
    [[ "$ext" != "$filename" ]] && ext=".$ext" || ext=""

    # Get the next available filename
    local new_filename
    new_filename=$(get_next_name "$base_dir" "$base_name" "$ext")

    # Rename the file and check for errors
    if mv "$original_file" "$new_filename"; then
        echo "Renamed $original_file to $new_filename"
    else
        echo "Error: Failed to rename $original_file to $new_filename"
        return 1
    fi
}

# Function to rename a list of files in the specified directory.
# Arguments:
#   $1 - Directory path where the files are located.
#   $@ - List of filenames to rename (e.g., "params.json", "train.sh").
# Behavior:
#   For each filename, parses the base name (without extension) and calls rename_file to rename it with incremented numbering.
rename_files() {
    local base_dir="$1"
    shift  # Shift past the first argument (base_dir)
    local files=("$@")  # Remaining arguments are the files to rename

    # Loop through each file and rename it
    for file in "${files[@]}"; do
        local base_name="${file%.*}"  # Extract base name without extension
        rename_file "$base_dir" "$file" "$base_name"
    done
}

# Function to rename a directory by incrementing its name to the next available number.
# Arguments:
#   $1 - Full path of the directory to be renamed.
# Behavior:
#   Extracts the parent directory and base name from the provided path,
#   then renames the directory to the next available name in the format "base_name_N",
#   where N is an incremented number.
#   Outputs the new directory path upon successful renaming.
rename_dir() {
    local dir_to_rename="$1"

    # Remove any trailing slash from the directory path
    dir_to_rename="${dir_to_rename%/}"

    # Check if the directory to rename exists
    if [[ ! -d "$dir_to_rename" ]]; then
        >&2 echo "Warning: Directory $dir_to_rename does not exist. Skipping renaming."
        return 1
    fi

    # Extract the parent directory path from the full directory path
    local parent_dir
    parent_dir=$(dirname "$dir_to_rename")

    # Extract the base name of the directory (the last component of the path)
    local base_name
    base_name=$(basename "$dir_to_rename")

    # Get the next available directory name in the parent directory
    # Example: If base_name is "run_info" and "run_info_1" exists,
    #          it will find "run_info_2" as the next available name
    local new_dir_name
    new_dir_name=$(get_next_name "$parent_dir" "$base_name" "")

    # Attempt to rename the directory to the new incremented name
    # This moves "dir_to_rename" to "new_dir_name"
    if mv "$dir_to_rename" "$new_dir_name"; then
        >&2 echo "Renamed directory $dir_to_rename to $new_dir_name"
        # Output the new directory path
        echo "$new_dir_name"
    else
        >&2 echo "Error: Failed to rename directory $dir_to_rename to $new_dir_name"
        return 1
    fi
}
