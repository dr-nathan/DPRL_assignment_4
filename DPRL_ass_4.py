import matplotlib.pyplot as plt
import numpy as np

# ------------------- 1.1 -------------------

# set up the environment
n_states = 5
disc = 0.9  # discount factor
# rewards
rew_a = np.array([0.2, 0, 0, 0, 1])
rew_b = np.array([0.2, 0, 0, 0, 0])
# actions
ac_a = np.array([1, 2, 3, 4, 4])
ac_b = np.array([0, 0, 0, 0, 0])

# empty Q table
Q_table = np.zeros((n_states, 2))  # states x actions
learning_rate = 0.5
converged = False
episode_steps = 100

total_steps = 0
while not converged:

    total_steps += 1
    # reset the environment
    state = 0
    # copy the Q table
    Q_table_old = np.copy(Q_table)

    # play episode
    for step in range(episode_steps):  # no terminal state, so just a fixed number of steps

        # choose action
        action = np.argmax(Q_table[state, :])  # can be 1 or 0
        # next state
        next_state = ac_a[state] if action == 1 else ac_b[state]
        # get reward
        reward = rew_a[state] if action == 1 else rew_b[state]
        # update Q table
        Q_table[state, action] = Q_table[state, action] + learning_rate * (
                reward + disc * np.max(Q_table[next_state, :]) - Q_table[state, action])
        # update state
        state = next_state

    # check convergence
    if np.allclose(Q_table, Q_table_old, atol=0.001):
        converged = True

print(f'{Q_table = }')
print(f'{total_steps = }')

# plot Q table heatmap
plt.figure()
plt.imshow(Q_table, cmap='magma')
plt.xticks([0, 1], ['A', 'B'])
plt.xlabel('Action')
plt.ylabel('State')
plt.title('Q table')
plt.colorbar()
plt.show()
print(f'{Q_table = }')
print(f'{total_steps = }')

# ------------------- 1.2 -------------------

# same, but with epsilon-greedy action selection

# empty Q table
Q_table = np.zeros((n_states, 2))  # states x actions
converged = False
epsilon = 0.2

total_steps = 0
while not converged:

    total_steps += 1
    # reset the environment
    state = 0
    # copy the Q table
    Q_table_old = np.copy(Q_table)

    # play episode
    for step in range(episode_steps):  # no terminal state, so just a fixed number of steps

        # choose action
        if np.random.rand() < epsilon:
            action = np.random.randint(2)
        else:
            action = np.argmax(Q_table[state, :])
        # next state
        next_state = ac_a[state] if action == 1 else ac_b[state]
        # get reward
        reward = rew_a[state] if action == 1 else rew_b[state]
        # update Q table
        Q_table[state, action] = Q_table[state, action] + learning_rate * (
                reward + disc * np.max(Q_table[next_state, :]) - Q_table[state, action])
        # update state
        state = next_state

    # check convergence
    if np.allclose(Q_table, Q_table_old, atol=0.001):
        converged = True

print(f'{Q_table = }')
print(f'{total_steps = }')

# plot Q table heatmap
plt.figure()
plt.imshow(Q_table, cmap='magma')
plt.xticks([0, 1], ['A', 'B'])
plt.xlabel('Action')
plt.ylabel('State')
plt.title('Q table')
plt.colorbar()
plt.show()
