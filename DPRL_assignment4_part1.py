import matplotlib.pyplot as plt
import numpy as np


def run_until_convergence(learning_rate, dynamic_RL, episode_steps, n_states,
                          disc, policy, epsilon=None):

    # rewards and actions, made dynamically based on the number of states
    rew_a = np.zeros(n_states)
    rew_a[-1] = 1
    rew_b = np.zeros(n_states)
    rew_b[0] = 0.2
    # actions
    ac_a = np.arange(1, n_states+1)
    ac_a[-1] = ac_a[-2]  # last action is stay in last state
    ac_b = np.zeros(n_states, dtype=int)

    # empty Q table
    Q_table = np.zeros((n_states, 2))
    Q_val_1 = []
    Q_val_5 = []

    total_steps = 0
    for t in range(5000):

        total_steps += 1
        # reset the environment
        state = 0

        # play episode
        for step in range(episode_steps):  # no terminal state, so just a fixed number of steps

            if dynamic_RL:
                learning_rate = 1 / (step + 1)

            # choose action
            if policy == 'greedy':
                # little trick to not always have the agent default to action A when the Q values are equal
                if Q_table[state, 0] == Q_table[state, 1]:
                    action = np.random.choice([0, 1])
                else:
                    action = np.argmax(Q_table[state, :])

            elif policy == 'epsilon_greedy':
                if np.random.random() < epsilon:
                    action = np.random.choice([0, 1])
                else:
                    if Q_table[state, 0] == Q_table[state, 1]:
                        action = np.random.choice([0, 1])
                    else:
                        action = np.argmax(Q_table[state, :])
            else:
                raise ValueError(f'Unknown policy: {policy}')

            # next state
            next_state = ac_a[state] if action == 0 else ac_b[state]
            # get reward
            reward = rew_a[state] if action == 0 else rew_b[state]

            # update Q table
            Q_table[state, action] = Q_table[state, action] + learning_rate * (
                    reward + disc * np.max(Q_table[next_state, :]) - Q_table[state, action])
            # update state and loop
            state = next_state

        # save Q values of state 1 and 5
        Q_val_1.append(Q_table[0, 1])
        Q_val_5.append(Q_table[-1, 0])

    print(f'{Q_table = }')
    print(f'{total_steps = }')

    return Q_table, Q_val_1, Q_val_5


def plot_actions(q_val1, q_val5):
    plt.plot(q_val1, label='Q(1, A)')
    plt.plot(q_val5, label='Q(5, B)')
    plt.xlabel('Episode')
    plt.ylabel('Q value')
    plt.title('Q values')
    plt.legend()
    plt.show()


def plot_policy_heatmap(Q_tab):
    plt.figure()
    plt.imshow(Q_tab, cmap='magma')
    plt.xticks([0, 1], ['A', 'B'])
    plt.xlabel('Action')
    plt.ylabel('State')
    plt.title('Q table')
    plt.colorbar()
    plt.show()


# ------------------- 1.1 -------------------

n_states = 5
disc = 0.9  # discount factor
policy = 'greedy'  # 'greedy' or 'epsilon-greedy'
episode_steps = 50
LR = 0.5
dynamic_RL = False  # if true, overrides the learning rate and uses 1/(t+1)
Q_table, val1, val5 = run_until_convergence(LR, dynamic_RL, episode_steps, n_states, disc, policy)

plot_actions(val1, val5)
plot_policy_heatmap(Q_table)

# ------------------- 1.2 -------------------

# same, but with epsilon-greedy action selection
epsilon = 0.55
Q_table, val1, val5 = run_until_convergence(LR, dynamic_RL, episode_steps, n_states, disc, 'epsilon_greedy', epsilon)

plot_actions(val1, val5)
plot_policy_heatmap(Q_table)

# ------------------- 1.3 -------------------
# with 10 actions
n_states = 10
epsilon = 1
Q_table, val1, val5 = run_until_convergence(LR, dynamic_RL, episode_steps, n_states, disc, 'epsilon_greedy', epsilon)

plot_actions(val1, val5)
plot_policy_heatmap(Q_table)

