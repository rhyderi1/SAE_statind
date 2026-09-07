# SAE_Statind ReadME
Project TLDR: We want to generalize the feature absorption pathology (introduced in Chanin et al., 2025) to modalities beyond language. Specifically, we want to be able to derive an absorption metric that requires minimal data assumptions. 

This github repo is roughly organized into 4 working folders. Note that some parts are a work in progress.
### 1. src/
This is where all the code is (.py mostly)
I've tried to include a brief description of each file at the beginning as a docstring (may be AI generated). Work in progress. 
### 2. scripts/
Shell scripts to run source code in src/ via sbatch. Includes job_name, output directory, GPU #, GPU memory, CPU number, job time, etc...
### 3. figures/
Roughly organized into folders based on figure type and SAE config
Files (.png) are mostly named descriptively with important parameters (layer, sae, sparsity, timestamp) included in filename
* There are many figures so this is still a work in progress
### 4. results/
Not used much, but intented for any results that are not text outputs or figures (eg .json files)

## How to use this repository
- Create a python (or other) script. For user input, argparse is used
- To run, enter the scripts/ directory, and choose one of the .sh files. They are all the same; there are multiple so that multiple scripts can be run in parallel.
- In the shell script, change the .py file at the bottom. Argparse arguments can be added sequentially using '--'
- Also change the job configuration (#GPUs, #CPUs, GPU memory, job name, time, etc...) to what you need
- Run the shell script using sbatch (preferred): eg. If you are running job.py on run_job3.sh, the terminal command is 'sbatch scripts/run_job3.sh'
- Check job progress with squeue -u {user}, and job time with sacct -j {job_id}
- Job text outputs are found in logs/{job_name}_{job_id}.out


  
