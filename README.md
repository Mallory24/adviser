Implemenation of Rank-based PER on ADvISER.    

ADvISER is a flexible dialog framework for research and education.[1]   
See their `documentation <https://digitalphonetics.github.io/adviser/>`for more details.  

Implementation
============

Add Rank-based Experience Replay Buffer
--------------------

* Rank-based experience replay is a prioritized method to make agent learn faster.    
* To efficiently sample and update priority in the buffer, a binary heap structure is used.  


Train & Test 
-----------------------

* Use integration tests to simulate user's behavior for trainning and evaluation.  
* Trainning is performed with the initialization of 10 random seeds, 10 epochs and 1000 episodes.  
* Evaluation is done after 10 trainning epochs, and is averaged with 500 episodes.   


Result
------------------------------

To evaluate agent's performance, 3 metrics are used:
1. Average Turns  
2. Average Success Rate 
3. Average Rewards (maximum = 20)

![train_rewards](https://github.com/Mallory24/adviser/blob/rank-based-PER-DQN/tests/integrationtests/logs/train_rewards.png)
![eval_turns](https://github.com/Mallory24/adviser/blob/rank-based-PER-DQN/tests/integrationtests/logs/eval_turns.png)
![eval_success](https://github.com/Mallory24/adviser/blob/rank-based-PER-DQN/tests/integrationtests/logs/eval_success.png)
![eval_rewards](https://github.com/Mallory24/adviser/blob/rank-based-PER-DQN/tests/integrationtests/logs/eval_rewards.png)


Reference
===========
[1]:![ref](https://github.com/Mallory24/adviser/blob/rank-based-PER-DQN/docs/ref.png)
