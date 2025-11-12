0. Fix validation loss on main, use the correct way of copying, avoid copying tensor
2. Figure out way of importing upto date LLM models. Can we still plug in hugging face models if needed?
4. Why is the slurm_scripts/submit.sh so complex? What is project_config for?  
5. How to handle data paths? Right now they are harded in two places: data/data_utils/get_data_dir and data/process_data/DATA_DIR


Questions for Nikhil:
1. Confused about priority of configs. In the scripts we have:
    lr=${1:-0.0001}
    wd=${2:-0.0}
    But this is already being set in the optimization config such as :
    adamw.json
    Is the learning rate 0.0001 from the script being used? Is it just a flag to say we should use lr?
    

Remove non hydra related code:
-  run

Suspect code for removing:
- load_model_and_tokenizer and  load_model_huggingface and not being used anywhere

Later:
- Rotary positional embeddings
- Implement MuonMax-Momo
- How best to keep/intergrate param_configs and scripts for submitting multiple jobs