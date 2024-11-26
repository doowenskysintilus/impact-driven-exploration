# Studying Sound Impact in Reinforcement Learning

## Installation

```
# create a new conda environment
conda create -n rlsound python=3.9
conda activate rlsound

# install dependencies
git clone 
cd impact-driven-exploration
pip install -r requirements.txt

# install MiniGrid
cd gym-minigrid
python setup.py install
```

## Checking if install works

```
python main.py --model vanilla --env MiniGrid-MultiRoom-N2-S4-v0 --total_frames 20000000 --num_actors 3 --savedir outputs/
python visu_agent.py --expe_path .\outputs\torchbeast-20241126-113529\ --env MiniGrid-MultiRoom-N2-S4-v0
```


## Acknowledgements
Our vanilla RL algorithm is based on [Torchbeast](https://github.com/facebookresearch/torchbeast), which is an open source implementation of IMPALA.

## License
This code is under the CC-BY-NC 4.0 (Attribution-NonCommercial 4.0 International) license.
