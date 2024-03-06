import numpy
import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
from collections import deque
import heapdict


class Node:
    def __init__(self, state=None, parent=None, action=-1, cost=0, hole=None, g=0, h=0, f=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.hole = hole
        self.g = g
        self.h = h
        self.f = f

    def expand_sons(self, env: DragonBallEnv) -> List:
        sons = []
        for action, successor in env.succ(self.state).items():
            hole = False
            if successor[0] is None:
                continue
            if successor[2] and not env.is_final_state(successor[0]):
                hole = True
            son = Node(successor[0], self, action, successor[1], hole)
            sons.append(son)
        return sons


class BFSAgent:
    def find_path(self, node: Node, expanded: int) -> Tuple[List[int], int, int]:
        total_cost = 0
        actions = []
        while node.parent is not None:
            total_cost += node.cost
            actions.append(node.action)
            node = node.parent
        actions.reverse()
        return actions, total_cost, expanded

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        # reset for new game !!!!!
        env.reset()

        # initialize DS
        expanded = 0
        start_node = Node(env.get_initial_state(), None)
        openq = deque()
        openq.append(start_node)
        closeq = deque()

        # check if empty
        if env.is_final_state(start_node.state):
            return [], 0, 0

        while len(openq) != 0:
            # pop new node
            node = openq.popleft()
            closeq.append(node.state)
            if env.is_final_state(node.state) is True:  # we got both dragons:
                return self.find_path(node, expanded)
            expanded += 1
            if node.hole:
                continue
            for action in range(4):
                env.reset()
                env.set_state(node.state)
                new_state, new_cost, new_terminated = env.step(action)
                if new_state is None:
                    continue
                hole = False
                if new_terminated and not env.is_final_state(new_state):
                    hole = True
                son = Node(new_state, node, action, new_cost, hole)
                if (son.state not in closeq) and (son.state not in {open_node.state for open_node in openq}):
                    if env.is_final_state(son.state) is True:  # we got both dragons:
                        return self.find_path(son, expanded)
                    openq.append(son)
        return [], 0, 0


class WeightedAStarAgent():
    def __init__(self) -> None:
        pass

    def find_path(self, node: Node, expanded: int) -> Tuple[List[int], int, int]:
        total_cost = 0
        actions = []
        while node.parent is not None:
            total_cost += node.cost
            actions.append(node.action)
            node = node.parent
        actions.reverse()
        return actions, total_cost, expanded

    def compute_heuristic(self, env: DragonBallEnv, state) -> int:

        x_s, y_s = env.to_row_col(state)

        distances = []

        # Compute distance to goal states
        for g in env.get_goal_states():
            x_g, y_g = env.to_row_col(g)
            delta_x = abs(x_g - x_s)
            delta_y = abs(y_g - y_s)
            dist_tmp = delta_x + delta_y
            distances.append(dist_tmp)
        if not env.collected_dragon_balls[0]:
            x_d1, y_d1 = env.to_row_col(env.d1)
            delta_x = abs(x_s - x_d1)
            delta_y = abs(y_d1 - y_s)
            dist_tmp = delta_x + delta_y
            distances.append(dist_tmp)
        if not env.collected_dragon_balls[1]:
            # compute dragon ball 2 dist to s
            x_d2, y_d2 = env.to_row_col(env.d2)
            delta_x = abs(x_s - x_d2)
            delta_y = abs(y_d2 - y_s)
            dist_tmp = delta_x + delta_y
            distances.append(dist_tmp)

        # Find minimum distance
        min_dist = min(distances)

        return min_dist

    def search(self, env: DragonBallEnv, h_weight: float) -> Tuple[List[int], int, int]:
        env.reset()
        expanded = 0
        # find h
        h = self.compute_heuristic(env, env.get_initial_state())
        # start node
        start_node = Node(env.get_initial_state(), None)
        start_node.parent = None
        # init g,h,f for the algo A*
        start_node.g = 0
        start_node.h = h
        start_node.f = h_weight * h + 1 * (1 - h_weight)

        if env.is_final_state(start_node.state):
            return [], 0, 0

        openq = heapdict.heapdict()
        openq[start_node] = (start_node.f, start_node.state)
        closeq = []
        while len(openq) != 0:
            node, val = openq.popitem()
            closeq.append(node)
            if env.is_final_state(node.state) is True:  # we got both dragons:
                return self.find_path(node, expanded)
            expanded += 1
            # print(node.state, node.h, node.g, node.f)
            if node.hole:
                continue
            # expand_list = node.expand_sons(env)
            for action in range(4):
                env.reset()
                env.set_state(node.state)
                hole = False
                new_state, new_cost, new_terminated = env.step(action)
                if new_state is None:
                    continue
                if new_terminated and not env.is_final_state(new_state):
                    hole = True
                h = self.compute_heuristic(env, new_state)
                son = Node(new_state, node, action, new_cost, hole)
                son.g = node.g + son.cost
                son.f = h_weight * h + (1 - h_weight) * son.g
                son.h = h

                if (son.state not in {close_node.state for close_node in closeq}) and (
                        son.state not in {open_node.state for open_node in openq}):
                    openq[son] = (son.f, son.state)

                elif son.state in {open_node.state for open_node in openq}:
                    for key in openq.keys():
                        if key.state == son.state and key.f > son.f:
                            del openq[key]
                            openq[son] = (son.f, son.state)
                            break
                else:
                    for key in closeq:
                        if key.state == son.state and key.f > son.f:
                            openq[son] = (son.f, son.state)
                            closeq.remove(key)
                            break
        return [], 0, 0


class AStarEpsilonAgent():
    def find_path(self, node: Node, expanded: int) -> Tuple[List[int], int, int]:
        total_cost = 0
        actions = []
        while node.parent is not None:
            total_cost += node.cost
            actions.append(node.action)
            node = node.parent
        actions.reverse()
        return actions, total_cost, expanded

    def compute_heuristic(self, env: DragonBallEnv, state) -> int:

        x_s, y_s = env.to_row_col(state)

        distances = []

        # Compute distance to goal states
        for g in env.get_goal_states():
            x_g, y_g = env.to_row_col(g)
            delta_x = abs(x_g - x_s)
            delta_y = abs(y_g - y_s)
            dist_tmp = delta_x + delta_y
            distances.append(dist_tmp)
        if not env.collected_dragon_balls[0]:
            x_d1, y_d1 = env.to_row_col(env.d1)
            delta_x = abs(x_s - x_d1)
            delta_y = abs(y_d1 - y_s)
            dist_tmp = delta_x + delta_y
            distances.append(dist_tmp)
        if not env.collected_dragon_balls[1]:
            # compute dragon ball 2 dist to s
            x_d2, y_d2 = env.to_row_col(env.d2)
            delta_x = abs(x_s - x_d2)
            delta_y = abs(y_d2 - y_s)
            dist_tmp = delta_x + delta_y
            distances.append(dist_tmp)

        # Find minimum distance
        min_dist = min(distances)

        return min_dist

    def __init__(self) -> None:
        pass

    def search(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        env.reset()
        expanded = 0
        # find h
        h = self.compute_heuristic(env, env.get_initial_state())
        # start node
        start_node = Node(env.get_initial_state(), None)
        start_node.g = 0
        start_node.h = h
        start_node.f = h

        if env.is_final_state(start_node.state):
            return [], 0, 0

        openq = heapdict.heapdict()
        openq[start_node] = (h, start_node.state)
        closeq = []
        while len(openq) != 0:
            node, curr_val = openq.peekitem()
            focal = heapdict.heapdict()  # for A epsilon algorithm
            for curr_node, val in openq.items():
                h_temp = curr_val[0]
                if val[0] <= (1 + epsilon) * h_temp:
                    focal[curr_node] = (curr_node.g, curr_node.state)
            node, value = focal.popitem()  # node with min g
            del openq[node]
            closeq.append(node)
            if env.is_final_state(node.state) is True:  # we got both dragons:
                return self.find_path(node, expanded)
            expanded += 1
            if node.hole:
                continue
            for action in range(4):
                env.reset()
                env.set_state(node.state)
                hole = False
                new_state, new_cost, new_terminated = env.step(action)
                if new_state is None:
                    continue
                if new_terminated and not env.is_final_state(new_state):
                    hole = True
                son = Node(new_state, node, action, new_cost, hole)
                son.g = node.g + son.cost
                son.h = self.compute_heuristic(env, son.state)
                son.f = son.h + son.g

                if (son.state not in {close_node.state for close_node in closeq}) and (
                        son.state not in {open_node.state for open_node in openq}):
                    openq[son] = (son.f, son.state)
                elif son.state in {open_node.state for open_node in openq}:
                    for curr in openq.keys():
                        if curr.state == son.state and curr.g > son.g:
                            del openq[curr]
                            openq[son] = (son.f, son.state)
                            break
                else:
                    for curr in closeq:
                        if curr.state == son.state and curr.g > son.g:
                            openq[son] = (son.f, son.state)
                            closeq.remove(curr)
                            break
        return [], 0, 0
