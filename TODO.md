0. Fix validation loss on main, use the correct way of copying, avoid copying tensor
2. Figure out way of importing upto date LLM models
3. move all data related code (process_data, process_sli_pajam10b) to a data directory
4. Why is the slurm_scripts/submit.sh so complex? What is project_config for?  
5. Get rid of multiple copies of data handling files. Keep only process_data. 
6. generate_task_file: Is used by slurm_scripts/submit.sh to generate multiple jobs on different GPUs, known as a tasks list.
7. Clean up gpt-opt/utils.py no longer need default config. Move get_data_dir to some data_utils and things to where they belong. Including data paths stuff!
8. Make very small test run, to see if code is broken
9. Figure out exactly which load_data/data stuff is being used
10. Currently only gpt_model is being used. Can we still plug in hugging face models if needed?

Removing non hydra related code:
- run_single, run

Suspect code for removing:
- load_model_and_tokenizer and  load_model_huggingface and not being used anywhere

Later:
- Rotary positional embeddings
- Implement MuonMax-Momo
- How best to keep/intergrate param_configs and scripts for submitting multiple jobs