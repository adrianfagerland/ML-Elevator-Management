import os

hours = 7
name = "elevator_"
job_file = '/home/wx350715/projects/ML-Elevator-Management/elevator_management/train.job'

# get run id
run = 1
for file in os.listdir("/work/wx350715/elevator_output/"):
    if(file.startswith(name) and file.endswith(".out")):
        run += 1
name = name + str(run)



batch_file = f"""
#!/usr/bin/zsh
#SBATCH --mem-per-cpu=2048M
#SBATCH --job-name={name}

#SBATCH --output=/work/wx350715/elevator_output/elevator_{name}.err
#SBATCH --error=/work/wx350715/elevator_output/elevator_{name}.err


#SBATCH --time={hours}:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1


export CONDA_ROOT=$HOME/miniconda3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
module load Python/3.10.4
conda activate elevator


cd /home/wx350715/projects/ML-Elevator-Management/elevator_management
python -m rl.training
""".format(hours=hours, name=name)


with open(job_file, 'w') as f:
    f.write(batch_file)

#os.system(f"sbatch {job_file}")
