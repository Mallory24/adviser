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


import numpy as np
import torch
import math
import random

from modules.policy.rl.common import DEVICE
from modules.policy.rl.binary_heap import BinaryHeap
from utils import common

class Buffer(object):
    """ Base class for experience replay buffers

    Initializes the memory, provides a print function for the memory contents 
    and a method to insert new items into the buffer.
    Sampling has to be implemented by child classes.

    """

    def __init__(self, buffer_size: int, batch_size: int, state_dim: int, 
                discount_gamma: float = 0.99):
        assert buffer_size >= batch_size, 'the buffer hast to be larger than the batch size'
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.discount_gamma = discount_gamma

        # construct memory
        self.mem_state = torch.empty(buffer_size, state_dim, dtype=torch.float, device=DEVICE)
        self.mem_action = torch.empty(buffer_size, 1, dtype=torch.long, device=DEVICE)
        self.mem_reward = torch.empty(buffer_size, 1, dtype=torch.float, device=DEVICE)
        self.mem_next_state = torch.empty(buffer_size, state_dim, dtype=torch.float, device=DEVICE)
        self.mem_terminal = torch.empty(buffer_size, 1, dtype=torch.float, device=DEVICE)

        self.write_pos = 0
        self.last_write_pos = 0
        self.buffer_count = 0

        self._reset()


    def _reset(self):
        """ Reset the state between consecutive dialogs

            Will be executed automatically after store with terminal=True was
            called.
        """

        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.episode_length = 0


    def store(self, state: torch.FloatTensor, action: torch.LongTensor, reward: float,
              terminal: bool = False):
        """ Store an experience of the form (s,a,r,s',t).

        Only needs the current state s (will construct transition to s'
        automatically).

        Args:
            state (torch.tensor): this turn's state tensor, or None if terminal = True
            action (torch.tensor): this turn's action index (int), or None if terminal = True
            reward (torch.tensor): this turn's reward (float)
            terminal (bool): indicates whether episode finished (boolean)
        """

        reward /= 20.0

        if isinstance(self.last_state, type(None)): #and terminal == False:
            # first turn of trajectory, don't record since s' is needed
            self.last_state = state
            self.last_action = action
            self.last_reward = reward
            return False
        else:
            if terminal == True:
                if self.episode_length > 0:
                    # update last state's reward and set it to terminal
                    self.mem_terminal[self.last_write_pos] = float(True)
                    self.mem_reward[self.last_write_pos] += reward
                self._reset()
                return False
            else:
                # in-between turn of trajectory: record
                self.mem_state[self.write_pos] = \
                                        self.last_state.clone().detach()
                self.mem_action[self.write_pos][0] = self.last_action
                self.mem_reward[self.write_pos][0] = self.last_reward
                self.mem_next_state[self.write_pos] = state.clone().detach()
                self.mem_terminal[self.write_pos] = float(False)

                # update last encountered state
                self.last_state = state.clone().detach()
                self.last_action = action
                self.last_reward = reward

                # update write index
                self.last_write_pos = self.write_pos
                self.write_pos = (self.write_pos + 1) % self.buffer_size
                if self.buffer_count < self.buffer_size:
                    self.buffer_count += 1

                self.episode_length += 1
                return True


    def print_contents(self, max_size: int = None):
        """ Print contents of the experience replay memory.
        
        Args:
            max_size (int): restrict the number of printed items to this number (if not None)
        """
        # how many entries to print
        print_items = len(self)
        if max_size is not None:
            print_items = min(print_items, max_size)

        print("# REPLAY BUFFER CAPACITY: ", self.buffer_size)
        print("# CURRENT ITEM COUNT", len(self))
        
        for i in range(print_items):
            print("entry ", i)
            print("  action", self.mem_action[i])
            print("  reward", self.mem_reward[i])
            print("  terminal", self.mem_terminal[i])
            print('---------')
            # TODO finish printing buffer (state, reward, actions, belief?)


    def __len__(self):
        """ Returns the number of items currently inside the buffer """
        return self.buffer_count


    def sample(self):
        """ Sample from buffer, has to be implemented by subclasses """
        raise NotImplementedError



class UniformBuffer(Buffer):
    """ Experience replay buffer with uniformly random sampling """

    def __init__(self, buffer_size: int, batch_size: int, state_dim: int,
                 discount_gamma: float = 0.99, sample_last_transition: bool = True):
        """
        Args:
            sample_last_transition (bool): if True, a batch will always include the most recent
                                           transition
                                           (see Sutton: A deeper look at experience replay)
        """
        super(UniformBuffer, self).__init__(buffer_size, batch_size, state_dim,
                                            discount_gamma=discount_gamma)
        print("  REPLAY MEMORY: Uniform")
        self.sample_last_transition = sample_last_transition


    def sample(self):
        """ Sample from buffer.
        
        Returns:
            states, actions, rewards, next states, terminal state indicator {0,1}, buffer indices,
            None
        """
        # create random indices
        data_indices = []
        if self.sample_last_transition:
            # include last transition (was at write - 1)
            # - see Sutton: A deeper look at experience replay
            if self.write_pos - 1 < 0:
                # last transition filled the capacity of the buffer
                data_indices = [self.buffer_size - 1]
            else:
                data_indices = [self.write_pos - 1]
        data_indices.extend([common.random.randint(0, self.buffer_count - 1) for i in
                             range(self.batch_size - int(self.sample_last_transition))])
        data_indices = torch.tensor(data_indices, dtype=torch.long, device=DEVICE)

        state_batch = self.mem_state.index_select(0, data_indices)
        action_batch = self.mem_action.index_select(0, data_indices)
        reward_batch = self.mem_reward.index_select(0, data_indices)
        next_state_batch = self.mem_next_state.index_select(0, data_indices)
        terminal_batch = self.mem_terminal.index_select(0, data_indices)

        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, \
                data_indices, None



class NaivePrioritizedBuffer(Buffer):
    """ Prioritized experience replay buffer.
    
    Assigns sampling probabilities dependent on TD-error of the transitions.
    """
    def __init__(self, buffer_size: int, batch_size: int, state_dim: int, 
                 sample_last_transition: bool = False,
                 regularisation: float = 0.00001, exponent: float = 0.6, beta: float = 0.4,
                 beta_frames=100000, discount_gamma: float = 0.99):
        super(NaivePrioritizedBuffer, self).__init__(buffer_size, batch_size, state_dim,
                                                     discount_gamma=discount_gamma)
        print("  REPLAY MEMORY: NAIVE Prioritized")

        self.probs = [0.0] * buffer_size
        self.regularisation = regularisation
        self.exponent = exponent
        self.beta = beta
        self.beta_frames = beta_frames
        self.beta_grad = (1 - self.beta) / self.beta_frames  # anneal beta over time
        self.max_p = 1.0
        self.sample_last_transition = sample_last_transition

    def _priority_to_probability(self, priority: float):
        """ Convert priority number to probability space (inside [0,1]) """
        return (priority + self.regularisation) ** self.exponent

    def store(self, state: torch.FloatTensor, action: torch.LongTensor, reward: float,
              terminal: bool = False):
        """ Store an experience of the form (s,a,r,s',t).

        Only needs the current state s (will construct transition to s'
        automatically).

        Newly added experience tuples will be assigned maximum priority.

        Args:
            state: this turn's state tensor, or None if terminal = True
            action: this turn's action index (int), or None if terminal = True
            reward: this turn's reward (float)
            terminal: indicates whether episode finished (boolean)
        """

        if super(NaivePrioritizedBuffer, self).store(state, action, reward, terminal=terminal):
            # create new tree node only if something new was added to the buffers
            self.probs[self.last_write_pos] = self._priority_to_probability(self.max_p)


    def update(self, idx: int, error: float):
        """ Update the priority of transition with index idx """
        p = self._priority_to_probability(error)
        if p > self.max_p:
            self.max_p = p
        self.probs[idx] = p


    def sample(self):
        """ Sample from buffer.
        
        Returns:
            states, actions, rewards, next states, terminal state indicator {0,1}, buffer indices,
            importance weights
        """
        batch_size = self.batch_size
        batch_write_pos = 0
        data_indices = torch.empty(self.batch_size, dtype=torch.long, device=DEVICE)
        probabilities = torch.empty(self.batch_size, dtype=torch.float, device=DEVICE)

        self.sample_last_transition = True
        p_normed = np.array(self.probs[:self.buffer_count]) / np.linalg.norm(
                                                            self.probs[:self.buffer_count], ord=1)
        indices = common.numpy.random.choice(list(range(self.buffer_count)), size=self.batch_size,
                                             p=p_normed)
        if self.sample_last_transition:
            # include last transition (was at tree.write - 1) 
            # -> see Sutton: A deeper look at experience replay
            data_indices[0] = self.last_write_pos
            probabilities[0] = self.probs[self.last_write_pos]
            # correct size of batch
            batch_size = batch_size - 1
            batch_write_pos += 1
    
        # TODO add option to sample each segment uniformly

        for i in range(batch_write_pos, self.batch_size):
            data_indices[i] = int(indices[i])
            probabilities[i] = self.probs[data_indices[i]]
       
        # assemble batch from data indices
        s_batch = self.mem_state.index_select(0, data_indices)
        a_batch = self.mem_action.index_select(0, data_indices)
        r_batch = self.mem_reward.index_select(0, data_indices)
        t_batch = self.mem_terminal.index_select(0, data_indices)
        s2_batch = self.mem_next_state.index_select(0, data_indices)

        # calculate importance sampling weights
        importance_weights = float(len(self)) * probabilities
        importance_weights = importance_weights.pow(-self.beta)
        importance_weights = importance_weights / importance_weights.max(dim=0)[0].item()

        return s_batch, a_batch, r_batch, s2_batch, t_batch, data_indices, \
               importance_weights.view(-1, 1)


class RankPrioritizedBuffer(Buffer):
    def __init__(self, buffer_size, batch_size, state_dim, discount_gamma=0.99):
        super(RankPrioritizedBuffer, self).__init__(buffer_size, batch_size, state_dim,
                                                    discount_gamma=discount_gamma)
        print("  REPLAY MEMORY: RANK Prioritized")
        self.exponent = 0.6
        self.beta = 0.4
        self.memory_size = buffer_size
        self.n_partitions = 400  # partition number n, split memory size to n part
        self.priority_queue = BinaryHeap(self.buffer_size, self.batch_size, self.n_partitions, self.exponent)

    def store(self, state, action, reward, terminal=False):
        if super(RankPrioritizedBuffer, self).store(state, action, reward, terminal=terminal):
            # add experience to the priority queue at the same time
            if not isinstance(self.last_state, type(None)):
                #rebalance binary heap whenever the memory buffer is full
                if self.write_pos < self.last_write_pos:
                    self.priority_queue.rebalance()
                # new experience always get the maximum priority
                priority = self.priority_queue.get_max_priority()
                self.priority_queue.update(priority, self.last_write_pos)

    def sample(self):
        dist_index = int(math.floor(len(self)/ self.memory_size * self.n_partitions))
        distribution = self.priority_queue.distributions[dist_index]
        data_indices = torch.empty(self.batch_size, dtype=torch.long,
                                    device=DEVICE)
        probabilities = torch.empty(self.batch_size, dtype=torch.float,
                                    device=DEVICE)
        rank_indices = []

        for x in distribution["ranges"]:
            sampled_object = random.sample(range(x[0],x[-1]+1), 1)
            rank_indices.append(sampled_object[0])
        e_id = self.priority_queue.get_experience_id(rank_indices)
        #print(e_id)

        for i in range(self.batch_size):
            data_indices[i] = int(e_id[i])
            probabilities[i] = distribution['pmf'][rank_indices[i]-1]

        # assemble batch from data indices
        s_batch = self.mem_state.index_select(0, data_indices)
        a_batch = self.mem_action.index_select(0, data_indices)
        r_batch = self.mem_reward.index_select(0, data_indices)
        t_batch = self.mem_terminal.index_select(0, data_indices)
        s2_batch = self.mem_next_state.index_select(0, data_indices)

        # calculate importance sampling weights
        # w = (P(i) * N) ^ (-beta) / max w
        importance_weights = float(len(self)) * probabilities
        importance_weights = importance_weights.pow(-self.beta)
        importance_weights = importance_weights / importance_weights.max(dim=0)[0].item()

        return s_batch, a_batch, r_batch, s2_batch, t_batch, data_indices, importance_weights.view(-1, 1)

    def update(self, e_id, td_error):
        self.priority_queue.update(abs(td_error), e_id)
