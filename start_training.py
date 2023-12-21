import os
import datetime
from pathlib import Path
import time
from itertools import product
networks = ["alpha"]
log_dir = Path("/work/wx350715/elevator_output")

prior_vars = set(dir())
prior_vars = set(dir())
# START PARAMETERS IN NAME STRING
hours = 2
max_training_steps = 3_000_000
update_timestep = 1000
K_epochs =  50
eps_clip =  0.2
gamma =     0.9
lr_actor =  0.003
lr_critic = 0.01
num_arrivals = 1000
density = 0.1
num_elevator_min = 1
num_elevator_max = 1
num_floors = 7
seed=2
# END PARAMETERS IN NAME STRING
new_names = set(dir()) - prior_vars

product_test = []
product_names = []

for new_var in new_names:
    var = globals()[new_var]
    if type(var) == list:
        product_test.append(var)
        product_names.append(new_var)


for configuration in product(*product_test):
    for idx, product_name in enumerate(product_names):
        globals()[product_name] = configuration[idx]
    message = ""
    for new_var in sorted(new_names):
        message += new_var + ":" + str(globals()[new_var]) + ","

    for network in sorted(networks):
        name = network + "_"
        message = network + ";" + message[:-1]
        job_file = '/home/wx350715/projects/ML-Elevator-Management/elevator_management/train.job'
        current_day = datetime.datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.datetime.now().strftime("%H-%M-%S")

        tmp_log_dir = log_dir/current_day/current_time

        print("Recorded Time:")
        print(current_day, current_time)
        os.makedirs(tmp_log_dir, exist_ok=True)



        batch_file = f"""#!/usr/bin/zsh
#SBATCH --mem-per-cpu=2048M
#SBATCH --job-name={name}

#SBATCH --output={tmp_log_dir}/out.out
#SBATCH --error={tmp_log_dir}/err.err


#SBATCH --time={hours}:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1


export CONDA_ROOT=$HOME/miniconda3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
module load Python/3.10.4
conda activate elevator


cd /home/wx350715/projects/ML-Elevator-Management/elevator_management
python -m rl.training --model {network}  --time_date {current_day}  --time_hours {current_time}  --message '{message}' --max_training_timesteps {max_training_steps} --update_timestep {update_timestep} --K_epochs {K_epochs} --eps_clip {eps_clip} --gamma {gamma} --lr_actor {lr_actor} --lr_critic {lr_critic} --random_seed {seed} --num_arrivals {num_arrivals} --num_elevators_min {num_elevator_min} --num_elevators_max {num_elevator_max} --num_floors {num_floors} --density {density}
"""

        with open(job_file, 'w') as f:
            f.write(batch_file)

        print(f"Run Job with name {name}")
        out = os.system(f"sbatch {job_file}")
        print(f"Job run " + "successfully" if out == 0 else "unsuccessfully")
        time.sleep(1)
        