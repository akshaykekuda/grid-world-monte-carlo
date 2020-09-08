"""Implementing Monte Carlo first visits and Monte Carlo Every Visits"""

import numpy as np
import random
import matplotlib.pyplot as plt

grid_row_len = 4
grid_col_len = 4
WIN_STATE = ((0, 0), (grid_row_len - 1, grid_col_len - 1))
reward = -1
gamma = 1

action_list = ['up', 'down', 'right', 'left']

optimal_policy = [['left'],
                  ['left'],
                  ['left', 'down'],
                  ['up'],
                  ['left', 'up'],
                  ['left', 'down'],
                  ['down'],
                  ['up'],
                  ['up', 'right'],
                  ['down', 'right'],
                  ['down'],
                  ['up', 'right'],
                  ['right'],
                  ['right']]


def state_tuple_converter(state_tuple):
    state = 4 * state_tuple[0] + state_tuple[1]
    return state


def state_converter(state):
    q = state // 4
    r = state % 4
    return q, r


def take_action(action, state_tuple):
    if action == 'up':
        return max(state_tuple[0] - 1, 0), state_tuple[1]
    if action == 'down':
        return min(state_tuple[0] + 1, grid_row_len - 1), state_tuple[1]
    if action == 'left':
        return state_tuple[0], max(state_tuple[1] - 1, 0)
    if action == 'right':
        return state_tuple[0], min(state_tuple[1] + 1, grid_col_len - 1)


def prob_action_select(action):
    if action == 'up':
        return np.random.choice(action_list, p=[0.7, 0.1, 0.1, 0.1])
    if action == 'down':
        return np.random.choice(action_list, p=[0.1, 0.7, 0.1, 0.1])
    if action == 'right':
        return np.random.choice(action_list, p=[0.1, 0.1, 0.7, 0.1])
    if action == 'left':
        return np.random.choice(action_list, p=[0.1, 0.1, 0.1, 0.7])


def generate_episode(state_tuple):
    generate_episode.state_visited.append(state_tuple)
    state = state_tuple_converter(state_tuple)
    if state_tuple == WIN_STATE[0] or state_tuple == WIN_STATE[1]:
        return False
    else:
        optimal_state_action = optimal_policy[state - 1]
        if len(optimal_state_action) == 1:
            prob_optimal_action = prob_action_select(optimal_state_action[0])
        else:
            prob_optimal_action = prob_action_select(np.random.choice(optimal_state_action, p=[0.5, 0.5]))

        next_state = take_action(prob_optimal_action, state_tuple)
        return generate_episode(next_state)


def compute_returns_mc_fv(episode):
    g_return = 0
    episode_len = len(episode) - 1
    for k in range(episode_len - 1, -1, -1):
        g_return = gamma * g_return + reward
        if not (episode[k] in episode[0:k]):
            return_dict_mc_fv.setdefault(state_tuple_converter(episode[k]), []).append(g_return)
    return return_dict_mc_fv


def compute_returns_mc_ev(episode):
    g_return = 0
    episode_len = len(episode) - 1
    for k in range(episode_len - 1, -1, -1):
        g_return = gamma * g_return + reward
        return_dict_mc_ev.setdefault(state_tuple_converter(episode[k]), []).append(g_return)
    return return_dict_mc_ev


def value_state_compute(dict_return):
    value_state = []
    for x in range(0, grid_row_len):
        for y in range(0, grid_col_len):
            curr_state = state_tuple_converter((x, y))
            value_state.append(round(np.mean(dict_return[curr_state]), 2))
    return value_state


def initialize_return_dict():
    return {0: 0,
                     1: [],
                     2: [],
                     3: [],
                     4: [],
                     5: [],
                     6: [],
                     7: [],
                     8: [],
                     9: [],
                     10: [],
                     11: [],
                     12: [],
                     13: [],
                     14: [],
                     15: 0}


def monte_carlo_episode(no_of_episodes):
    print("Number of Episodes=" + str(no_of_episodes))
    for i in range(0, no_of_episodes):
        start_state = random.randint(1, 14)
        start_state_tuple = state_converter(start_state)
        generate_episode.state_visited = []
        generate_episode(start_state_tuple)
        return_mc_first_visit = compute_returns_mc_fv(generate_episode.state_visited)
        return_mc_every_visit = compute_returns_mc_ev(generate_episode.state_visited)

    return value_state_compute(return_mc_first_visit), value_state_compute(return_mc_every_visit)


return_dict_mc_fv = initialize_return_dict()
return_dict_mc_ev = initialize_return_dict()
fv, ev = monte_carlo_episode(100)
print("FV = " + str(fv))
print("EV = " + str(ev))
plt.subplot(2, 2, 1)
plt.title("No. of Episodes = 100")
plt.plot(fv, 'o', label="MC First Visit")
plt.plot(ev, 'o', label="MC Every Visit")
plt.xlabel('State')
plt.ylabel('Value of state')
plt.xticks(list(range(16)))
plt.legend()

return_dict_mc_fv = initialize_return_dict()
return_dict_mc_ev = initialize_return_dict()
fv, ev = monte_carlo_episode(1000)
print("FV = " + str(fv))
print("EV = " + str(ev))
plt.subplot(2, 2, 2)
plt.title("No. of Episodes = 1000")
plt.plot(fv, 'o', label="MC First Visit")
plt.plot(ev, 'o', label="MC Every Visit")
plt.xlabel('State')
plt.ylabel('Value of state')
plt.xticks(list(range(16)))
plt.legend()

return_dict_mc_fv = initialize_return_dict()
return_dict_mc_ev = initialize_return_dict()
fv, ev = monte_carlo_episode(10000)
print("FV = " + str(fv))
print("EV = " + str(ev))
plt.subplot(2, 2, 3)
plt.title("No. of Episodes = 10000")
plt.plot(fv, 'o', label="MC First Visit")
plt.plot(ev, 'o', label="MC Every Visit")
plt.xlabel('State')
plt.ylabel('Value of state')
plt.xticks(list(range(16)))
plt.legend()

return_dict_mc_fv = initialize_return_dict()
return_dict_mc_ev = initialize_return_dict()
fv, ev = monte_carlo_episode(100000)
print("FV = " + str(fv))
print("EV = " + str(ev))
plt.subplot(2, 2, 4)
plt.title("No. of Episodes = 100000")
plt.plot(fv, 'o', label="MC First Visit")
plt.plot(ev, 'o', label="MC Every Visit")
plt.xlabel('State')
plt.ylabel('Value of state')
plt.xticks(list(range(16)))
plt.legend()

plt.figtext(0.01, 0.03, "Note1: Value of discount factor, gamma ="+str(gamma), bbox=dict(facecolor='red', alpha=0.5))
plt.figtext(0.78, 0.03, "Note2: States 0 and 15 are goal states", bbox=dict(facecolor='red', alpha=0.5))

plt.subplots_adjust(top=0.95, hspace=0.3)
plt.show()
