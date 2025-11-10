0. Fix validation loss on main, use the correct way of copying, avoid copying tensor
2. Figure out way of importing upto date LLM models
3. move all data related code (process_data, process_sli_pajam10b) to a data directory
4. Why is the slurm_scripts/submit.sh so complex? What is project_config for?  
5. Get rid of multiple copies of data handling files. Keep only process_data. 
6. generate_task_file: Is used by slurm_scripts/submit.sh to generate multiple jobs on different GPUs, known as a tasks list.
7. Clean up gpt-opt/utils.py no longer need default config. Move things to where they belong. Including data paths stuff!


Removing non hydra related code:
- utils.py/get_default_config
- run_single, run

Later:
- Rotary positional embeddings
- Implement MuonMax-Momo
- How best to keep/intergrate param_configs and scripts for submitting multiple jobs