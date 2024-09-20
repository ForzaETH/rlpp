# The PBL F1TENTH Gym Environment

This is the repository for the PBL F1TENTH Gym environment.

The original project is still under heavy development, and this project branched off from it, so do not expect the code to be coherent overall.

You can find the [documentation](https://f1tenth-gym.readthedocs.io/en/latest/) for the original F1TENTH environment here.

## Quickstart
(Basically the same as for the original environment)

Please ensure that your Python version is 3.8!!!

Note: it is suggested to use a virtualenvironment of some kind

You can install the environment by running:
```bash
git clone pbl-f1tenth-gym   # link to be updated very soon
cd ~/pbl-f1tenth-gym
pip install -r requirements.txt
pip install -e .
```

Install the splinify-package and the modified stable-baselines-3:
```bash
git stable-baselines-3   # link to be updated very soon
cd ~/stable-baselines-3
pip install -e .
git clone splinify-package   # link to be updated very soon
cd ~/splinify-package
pip install .
```

Reinstall the correct version of pyglet because something installs it to an old version:
```bash
cd ~/pbl-f1tenth-gym
pip install -r requirements.txt
```
TODO: fix this, prob in splinify
Install the splinify-package and the modified stable-baselines-3.

## Import New Maps

Copy the entire map folder from the racestack into the `custom_maps/map_imported` folder.

Then run the following command:

```bash
cd custom_maps
python3 import_ROS_map.py
```

Finally, go to `wandb_train.py` and modify `wandb_config.yaml`, changing the `map_name` to the new map name.

Now training should be conducted on the new map.

Sure! Here is the revised "Training Configuration" section in a list format:

## Training Configuration

All configuration YAML files are inside the `/wandb_trains` folder. Here's a summary of each configuration file:

1. **wandb_config.yaml**:
   - Contains the most important parameters for the gym and training.

2. **nn_config.yaml**:
   - Defines the training length.

3. **recorder_config.yaml**:
   - Should be updated automatically and does not need to be modified manually.

4. **plot_config.yaml**:
   - Provides tuning options for plotting after training or testing.
   - Modify it if you are not satisfied with the standard summary plots.

5. **search_space.yaml**:
   - Enables grid search with certain parameters.
   - Feel free to add more parameters if needed.
   - A number of experiments with different parameters will be executed one by one automatically.
   - Note: It is better to perform one parameter search at a time, as having too many experiments running in a row could lead to a crash.



## Modify the Gym Dynamical Parameters

The physical parameters regarding the car and dynamic models can be found inside the `/gym/f110_gym/envs` folder. The `SIM_pacejka.yaml` and `SIM_car_params.yaml` are the two default enabled settings.

## Observation Space

This gym environment has a quite different observation space compared to the original F1TENTH one. As it is mainly made for RL purposes, it is a single array with a subset of the states chosen and normalized between zero and one.

There are two observation modes. To choose one, you should do the following:
```python
gym.make('f110_gym:f110-v0', obs_mode='frenet', ...)
``` 
or
```python
gym.make('f110_gym:f110-v0', obs_mode='trajectory_frenet', ...)
```

The first observation, `frenet`, will return: 
- Lateral deviation from a trajectory
- Relative heading to a trajectory 
- Longitudinal velocity of the car
- Lateral velocity of the car
- Yaw rate

The second observation, `trajectory_frenet`, will add an array of points consisting of a piece of the reference trajectory and the track boundary in front of the car. 

These observations are, however, normalized. Therefore, for more control-oriented usage, the following ways of interfacing are preferred:
- **Option one**: using the state of the simulator directly, which can be accessed through `env.agents[0].state` (if using one car) and consists of:
  - x position in global coordinates [m]
  - y position in global coordinates [m]
  - Steering angle of front wheels [rad]
  - Velocity in x direction [m/s]
  - Yaw angle [rad]
  - Yaw rate [rad/s]
  - Slip angle at vehicle center [rad]
- **Option two**: using the intermediate frenet representation, accessible through the attribute `env.observator.denorm_obs`, consisting of a dictionary with the following keys:
  - `deviation`: deviation from the reference trajectory [m]
  - `rel_heading`: heading relative to the reference trajectory [rad]
  - `longitudinal_vel`: longitudinal velocity of the car [m/s]
  - `later_vel`: lateral velocity of the car [m/s]
  - `yaw_rate`: yaw rate of the car [rad/s]

## Action Space 

The action space is currently composed of `steering` and `acceleration`, and the delay on the steering input can be customized.

## Known Issues

- Library support issues on Windows. You must use Python 3.8 as of 10-2021.
- On macOS Big Sur and above, when rendering is turned on, you might encounter the error:
```bash
ImportError: Can't find framework /System/Library/Frameworks/OpenGL.framework.
```
You can fix the error by installing a newer version of pyglet:
```bash
pip3 install pyglet==1.5.11
```
You might see an error similar to:
```bash
gym 0.17.3 requires pyglet<=1.5.0,>=1.4.0, but you'll have pyglet 1.5.11 which is incompatible.
```
which can be ignored. The environment should still work without error.

## Citing

If you find this Gym environment useful, please consider citing:

```bibtex
@inproceedings{okelly2020f1tenth,
  title={F1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning},
  author={O’Kelly, Matthew and Zheng, Hongrui and Karthik, Dhruv and Mangharam, Rahul},
  booktitle={NeurIPS 2019 Competition and Demonstration Track},
  pages={77--89},
  year={2020},
  organization={PMLR}
}
```