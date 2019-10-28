Implemenation of Rank-based PER on ADvISER.    

ADvISER is a flexible dialog framework for research and education.[1]   
Please see their `documentation <https://digitalphonetics.github.io/adviser/>`_ for more details.  

Implementation
============

Add Rank-based Experience Replay Buffer
--------------------

Rank-based experience replay is a prioritized method to make agent learn faster.    
To efficiently sample and update priority in the buffer, a binary heap structure is used.  


Train & Test 
-----------------------

Use integration tests to simulate user's behavior for trainning and evaluation.  
Trainning is performed with the initialization of 10 random seeds, 10 epochs and 1000 episodes.  
Evaluation is done after 10 trainning epochs, and is averaged with 500 episodes.   


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

1. Make sure you have virtualenv installed by executing

.. code-block:: bash

    python3 -m pip install --user virtualenv

2. Create the virtual environment (replace envname with a name of your choice)

.. code-block:: bash

    python3 -m venv <path-to-env>

3. Source the environment (this has to be repeated every time you want to use ADVISER inside a
new terminal session)

.. code-block:: bash

    source <path-to-env>/bin/activate

4. Install the required packages

.. code-block:: bash

    pip install -r requirements.txt

5. To make sure your installation is working, navigate to the adviser folder:

.. code-block:: bash

    cd adviser

and execute

.. code-block:: bash

    python run_chat.py --domain courses

6. Select a language by entering `english` or `german`, then chat with ADvISER. To end your
conversation, type `bye`.


Reference
===========
![ref](https://github.com/Mallory24/adviser/blob/rank-based-PER-DQN/docs/ref.png) [1]
