import sys
import math
import numpy as np

class BinaryHeap(object):

    def __init__(self, priority_size, batch_size, n_partitions, exponent, replace=True):
        self.max_size = priority_size
        self.queue_size = 0
        self.priority_queue = {}  #{1: (td-error, e_id1), 2: (td-error, e_id2) }
        self.rank_to_experience = {}
        self.k = batch_size
        self.n_partitions = n_partitions
        self.exponent = exponent
        self.replace = replace
        self.distributions = self.build_distributions()

    def check_cross(self, segment): #check which segment should include the object from previous segment
        segment = list(segment)
        cross = []
        for i in range(len(segment)):
            if i == (len(segment)-1):
                break
            elif segment[i] != segment[i+1]:
                cross.append(int(segment[i] + 1))
        cross = np.asarray(cross)
        return cross

    def build_distributions(self):
        distributions = {}
        partition_id = 1
        # each partition size
        partition_size = int(math.floor(self.max_size / self.n_partitions))

        for n in range(partition_size, self.max_size + 1, partition_size):
            distribution = {}
            # P(i) = (rank i) ^ (-alpha) / sum ((rank i) ^ (-alpha))
            pmf = np.array([(1 / i) ** self.exponent for i in range(1, n + 1)])
            pmf_norm = pmf / np.linalg.norm(pmf, ord=1)
            distribution['pmf'] = pmf_norm
            # print(distribution)
            cdf = np.cumsum(pmf_norm)
            # print(cdf)
            boundary = cdf[-1] / self.k
            segment = cdf // boundary  # which of the segment does each P(i) fall into
            cross_segment = self.check_cross(segment)
            # print(segment)
            # print(cross_segment)

            ranges = []  # list of tuples (left,right) that are inclusive ranges of rank\
            prev_range = (1, 1)
            for i in range(self.k):
                seg_point = np.nonzero(segment == (i + 1))[0]
                cross_point = np.nonzero(cross_segment == (i + 1))[0]
                if len(seg_point) > 0:
                    if len(cross_point) > 0:  # if the segment point is also where the cross point locates at
                        this_range = (
                        prev_range[-1], seg_point[-1] + 1)  # include the last object from previous range to this range
                        ranges.append(this_range)
                        prev_range = this_range
                    else:
                        this_range = (seg_point[0] + 1, seg_point[-1] + 1)

                        ranges.append(this_range)
                else:
                    if len(cross_point) > 0:  # if it is only a cross point, range is from previous one to the next one
                        this_range = (prev_range[-1], prev_range[-1] + 1)
                        ranges.append(this_range)
                        prev_range = (
                        prev_range[-1] + 1, prev_range[-1] + 1)  # replace the previous range with the next range index
                    else:
                        ranges.append(prev_range)
            # print(ranges)
            distribution["ranges"] = ranges
            distributions[partition_id] = distribution
            partition_id += 1
        return distributions

    def isFull(self):
        if self.queue_size > self.max_size:
            return True

    def get_max_priority(self):
        if self.queue_size == 0: #no entry before
            return 1
        else:
            return self.priority_queue[1][0] #always get the top priority

    def add(self, priority, e_id):
        self.queue_size += 1
        if self.isFull() and self.replace == False:
            sys.exit('Error: priority queue is full and replace is set to FALSE!\n')
            return False
        self.tmp_rank = min(self.queue_size, self.max_size)

        self.priority_queue[self.tmp_rank] = (priority, e_id)
        self.rank_to_experience[self.tmp_rank] = e_id
        self.up_heap(self.tmp_rank)
        return True

    def update(self, priority, e_id):
        if e_id in self.rank_to_experience.values():  # old experience, do update
            inv = {v: k for k, v in self.rank_to_experience.items()}
            rank_id = inv[e_id]
            self.priority_queue[rank_id] = (priority, e_id)
            self.rank_to_experience[rank_id] = e_id

            self.down_heap(rank_id)
            self.up_heap(rank_id)
            return True
        else:  # new experience, do insert
            return self.add(priority, e_id)

    def up_heap(self, node):
        if node > 1:
            parent = node // 2
            if self.priority_queue[node][0] >= self.priority_queue[parent][0]:
                tmp = self.priority_queue[parent]
                self.priority_queue[parent] = self.priority_queue[node]
                self.priority_queue[node] = tmp
                # change rank_to_experience
                self.rank_to_experience[parent] = self.priority_queue[parent][1]
                self.rank_to_experience[node] = self.priority_queue[node][1]
                self.up_heap(parent)

    def down_heap(self, node):
        if node < self.queue_size:
            biggest = node
            left = node * 2
            right = node * 2 + 1
            if left < self.queue_size and self.priority_queue[node][0] < self.priority_queue[left][0]:
                biggest = left
            if right < self.queue_size and self.priority_queue[node][0] < self.priority_queue[right][0]:
                biggest = right

            if biggest != node:
                tmp = self.priority_queue[biggest]
                self.priority_queue[biggest] = self.priority_queue[node]
                self.priority_queue[node] = tmp
                # change rank_to_experience
                self.rank_to_experience[biggest] = self.priority_queue[biggest][1]
                self.rank_to_experience[node] = self.priority_queue[node][1]
                self.down_heap(biggest)

    def rebalance(self, full=False):
        sort_array = sorted(self.priority_queue.values(), key=lambda x: x[0], reverse=True)
        if full:
            sorted_list = sorted(self.priority_queue.values(), key=lambda x: x[0], reverse=True)
            self.priority_queue.clear()
            self.rank_to_experience.clear()
            rank = 1
            while rank <= self.queue_size:
                self.priority_queue[rank] = sorted_list[rank - 1]
                self.rank_to_experience[rank] = sorted_list[rank - 1][1]
                rank += 1
            for i in range(int(self.queue_size // 2), 1, -1):
                self.down_heap(i)

    def get_experience_id(self, rank_ids):
        return [self.rank_to_experience[i] for i in rank_ids]
