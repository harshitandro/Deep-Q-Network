## Deep Q Networks

Generic implementation of Deep Q Networks using Keras for the purpose of solving RL control tasks having discrete action space.

### Requirements :
```
numpy
matplotlib
keras
imageio
gym (OpenAI Gym)
```

### Help :
```
$ python main_driver -h
```

### Environments :
List of OpenAI Gym Environments which are solved by agents in this repo: 
- #### CartPole-v0
    **Targeted Reward** (over 100 consecutive itrs) : >= 195 <br>
    **Reward Achieved** (over 100 consecutive itrs) : 200.0 <br>
    **Episodes before solve :**  265<br>
    
    Following are the commands used to train & test the model(include the necessary hyperparameters):
    - *Command to train :* ```python3 main_driver,py train CartPole-v0 --itr-count 2000 -b 64 -lr 0.0001 --gamma 0.99 --render False --avg-expected-reward -110.0```
    
    - *Command to test :* ```python3 main_driver,py test CartPole-v0 --render True --avg-expected-reward 195 --test-total-iterations 100 --model-path "best_save/CartPole-v0_local_model_1534607382.5808585.h5"```
    
    **Result Log Files:**
    - Training : [result_train_CartPole-v0_1534607382.5808585.log](results/result_train_CartPole-v0_1534607382.5808585.log)
    - Testing : [result_test_CartPole-v0_1534607382.5808585.log](results/result_test_CartPole-v0_1534607382.5808585.log)
    
    **Detailed Log Files:**
    - Training : [train_CartPole-v0_1534607382.5808585.log](logs/train_CartPole-v0_1534607382.5808585.log)
    - Testing : [test_CartPole-v0_1534607382.5808585.log](logs/test_CartPole-v0_1534607382.5808585.log)
    
    *Reward Plot:*<br>
    ![Reward Plot](results/reward_plot/reward_plot_CartPole-v0_1534607382.5808585.jpeg "Reward Plot")
    
    *Agent Playing the Env:*<br>
    ![Reward Plot](results/agent_play/agentplay_CartPole-v0_1534607382.5808585.gif "Agent Playing in the environment.")