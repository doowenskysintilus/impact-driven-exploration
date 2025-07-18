# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from src.arguments import parser 

from src.algos.torchbeast import train as train_vanilla 
from src.algos.count import train as train_count
from src.algos.curiosity import train as train_curiosity 
from src.algos.rnd import train as train_rnd
#from src.algos.ride import train as train_ride
from src.algos.she import train as train_she


from src.algos.no_episodic_counts import train as train_no_episodic_counts
from src.algos.only_episodic_counts import train as train_only_episodic_counts

def main(flags):
    if flags.model == 'vanilla':
        train_vanilla(flags)
    elif flags.model == 'count':
        train_count(flags)
    elif flags.model == 'curiosity':
        train_curiosity(flags)
    elif flags.model == 'rnd':
        train_rnd(flags)
    #elif flags.model == 'ride':
        #train_ride(flags)
    elif flags.model == 'she':
        
        train_she(flags)
    elif flags.model == 'no-episodic-counts':
        train_no_episodic_counts(flags)
    elif flags.model == 'only-episodic-counts':
        train_only_episodic_counts(flags)
    else:
        raise NotImplementedError("This model has not been implemented. "\
        "The available options are: vanilla, count, curiosity, rnd, ride, \
        no-episodic-counts, and only-episodic-count.")

if __name__ == '__main__':
    flags = parser.parse_args()
    main(flags)

#python main.py --model vanilla --env MiniGrid-FetchEnvSoundS8N3-v0 --total_frames 1000000 --num_actors 3 --savedir outputs/ 
#python visu_agent.py --expe_path outputs/torchbeast-20250122-111049 --env MiniGrid-FetchEnvSoundS8N3-v0
#python visu_agent.py --expe_path outputs/torchbeast-20250120-155603 --env MiniGrid-MultiRoom-N2-S4-v0
#python main.py --model ride --env MiniGrid-MultiRoom-N7-S4-v0 --total_frames 30000000 --intrinsic_reward_coef 0.1 --entropy_cost 0.0005
# python main.py --model she --env MiniGrid-FetchEnvSoundS8N3-v0 --total_frames 1000000 --intrinsic_reward_coef 0.1 --entropy_cost 0.0005
# python main.py --model she --env MiniGrid-FetchEnvSoundS8N3-v0 --total_frames 30000000 --intrinsic_reward_coef 0.1 --entropy_cost 0.0005 --no_reward
# oarsub -q default -p grouille -t exotic -I -l walltime=07:00:00
# ulimit -n 8192


