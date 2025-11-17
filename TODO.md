0. Fix validation loss on main, use the correct way of copying, avoid copying tensor
2. Figure out way of importing upto date LLM models. Can we still plug in hugging face models if needed?
4. Why is the slurm_scripts/submit.sh so complex? What is project_config for?  
5. How to handle data paths? Right now they are harded in two places: data/data_utils/get_data_dir and data/process_data/DATA_DIR


Questions for Nikhil:
1.  
    
For Monster Experiments
1. Figure out how to set up automatic 2nd sweep, using the first sweep to set the ratio.
2. Figure out how to plot outputs

Remove non hydra related code:
-  Done

Suspect code for removing:
- load_model_and_tokenizer  

Later:
- How best to keep/intergrate param_configs and scripts for submitting multiple jobs