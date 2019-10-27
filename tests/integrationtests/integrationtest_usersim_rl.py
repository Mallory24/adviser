###############################################################################
#
# Copyright 2019, University of Stuttgart: Institute for Natural Language Processing (IMS)
#
# This file is part of Adviser.
# Adviser is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3.
#
# Adviser is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Adviser.  If not, see <https://www.gnu.org/licenses/>.
#
###############################################################################

import os
import sys
#from statistics import mean
import random
import matplotlib.pyplot as plt

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(get_root_dir())


from dialogsystem import DialogSystem
from modules.bst import HandcraftedBST
from modules.simulator import HandcraftedUserSimulator
from modules.policy import DQNPolicy
from modules.policy.rl.experience_buffer import RankPrioritizedBuffer
from modules.policy.evaluation import PolicyEvaluator
from utils.domain.jsonlookupdomain import JSONLookupDomain
from utils import DiasysLogger, LogLevel


from utils import common

if __name__ == "__main__":
    logger = DiasysLogger(console_log
    _lvl=LogLevel.RESULTS, file_log_lvl=LogLevel.RESULTS)
    common.init_random()

    TRAIN_EPISODES = 1000
    NUM_TEST_SEEDS = 10
    EVAL_EPISODES = 500
    MAX_TURNS = -1
    TRAIN_EPOCHS = 10

    random_seeds = []
    for i in range(NUM_TEST_SEEDS):
        random_seeds.append(random.randint(0, 2 ** 32 - 1))  # add seed here if wanted

    results = {}
    for seed in random_seeds:
        common.init_once = False
        common.init_random(seed=seed)
        domain = JSONLookupDomain('ImsCourses')
        bst = HandcraftedBST(domain=domain, logger=logger)
        user_simulator = HandcraftedUserSimulator(domain=domain, logger=logger)
        policy_rank_PER = DQNPolicy(domain=domain, lr=0.0001, eps_start=0.3, gradient_clipping=5.0,
                           buffer_cls=RankPrioritizedBuffer, replay_buffer_size=20000, shared_layer_sizes=[256],
                           train_dialogs=TRAIN_EPISODES, target_update_rate=3, training_frequency=2, logger=logger)

        evaluator = PolicyEvaluator(domain=domain, use_tensorboard=True,
                                experiment_name='eval_rank_PER' + str(seed), logger=logger)
        ds = DialogSystem(policy_rank_PER,
                        user_simulator,
                        bst,
                        evaluator
                            )

        for i in range(TRAIN_EPOCHS):
            ds.train()
            evaluator.start_epoch()
            for episode in range(TRAIN_EPISODES):
                ds.run_dialog(max_length=MAX_TURNS)
            evaluator.end_epoch()

            ds.num_dialogs = 0  # IMPORTANT for epsilon scheduler in dqnpolicy
            policy_rank_PER.save()

        ds.eval()
        evaluator.start_epoch()
        for episode in range(EVAL_EPISODES):
            ds.run_dialog(max_length=MAX_TURNS)
        evaluator.end_epoch()

        results[seed] = {}
        results[seed]['eval_dialogs'] = evaluator.epoch_eval_dialogs
        results[seed]['eval_avg_turns'] = sum(evaluator.eval_turns) / evaluator.epoch_eval_dialogs
        results[seed]['eval_avg_success'] = sum(evaluator.eval_success) / evaluator.epoch_eval_dialogs
        results[seed]['eval_avg_reward'] = sum(evaluator.eval_rewards) / evaluator.epoch_eval_dialogs

        eval_avg_turns = [results[seed]['eval_avg_turns'] for seed in random_seeds]
        eval_avg_success = [results[seed]['eval_avg_success'] for seed in random_seeds]
        eval_avg_reward = [results[seed]['eval_avg_reward'] for seed in random_seeds]



    logger.result("")
    logger.result("###################################################")
    logger.result("")
    logger.result(" ### Eval with " + str(NUM_TEST_SEEDS) + " random seeds ###")
    logger.result(str(random_seeds))
    logger.result("# Num Dialogs " + str(EVAL_EPISODES))
    logger.result("# Avg Turns " + str(sum([results[seed]['eval_avg_turns'] for seed in random_seeds]) / NUM_TEST_SEEDS))
    logger.result("# Avg Success " + str(sum([results[seed]['eval_avg_success'] for seed in random_seeds]) / NUM_TEST_SEEDS))
    logger.result("# Avg Reward " + str(sum([results[seed]['eval_avg_reward'] for seed in random_seeds]) / NUM_TEST_SEEDS))


