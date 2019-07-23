## PyBrainGym -- OpenAI's Gym interface for PyBrain reinforcement learning algorithms

Library contains various classes served as interface between Gym and PyBrain libraries. 
It allows to use PyBrain reinforcement learning algorithms on Gym's environments. In 
addition it allows to execute experiments in parallel fashion reducing learning time.


### Features

* execution of Gym environments in PyBrain,
* execution of multiple experiments in parallel manner,
* custom transformation of data passed between Gym and PyBrain,
* quantization/digitization of continuous space (floating point) values and arrays of 
values to discrete number of states (integers),


### Examples

In directory *src/examples* there are following examples:
* *frozenlake/frozen.py* -- solution for *FrozenLake-v0* problem using *SARSA* learner.
* *cartpole/cart.py* -- solution for *CartPole-v1* problem using *SARSA* learner. It 
demonstrates how to discretize continuous input.
* *mountaincar* -- solution for discrete and continuous version of *MountainCar* problem 
using *SARSA* learner. In addition, continuous version run experiments in parallel manner. 
* *multiprocess* -- example of use of *multiprocessing* library.


### Running

1. checkout the project
2. execute *installenv.sh* script to install/configure virtual environment
3. execute *startenv.sh* to activate virtual environemnt 
4. execute *installpackage.sh* to install *pybraingym* library in development mode
5. execute *src/runtests.py* to run unit tests


### Requirements

Library requires *fixed* version of PyBrain library to work. It can be checked out from 
following address: [https://github.com/anetczuk/pybrain](https://github.com/anetczuk/pybrain)


### Known issues

*Q/SARSA* learner does not handle well last reward of episodic environments causing last 
reward and state to be ignored. It can be crucial in case when non-zero reward 
is only received when reaching goal.
As workaround, *processLastReward* function was introduced to call experiment *fake* 
iteration to allow agent to store the data.
