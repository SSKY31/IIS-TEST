# Intelligent Systems Project Base Code testing versions

## version: IIS-v2
base changes made for multi agent simulation with obstacles. No change in rewards and EnvWrapper

## Initialization
Initialization for static obstacle and dynamic obstacle is in 'init_map()' of 'rccar_env.py'
- change 'static_prob' and 'dynamic_dir'

## test.py
- code for testing PID control
- change 'num_controlled_agents', 'num_static_agents', 'num_dynamic_agents' values in args for different simulations
