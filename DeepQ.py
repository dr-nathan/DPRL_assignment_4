import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# NN with 1 hidden layer of 16 neurons, ReLU activation
class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 16)
        self.fc2 = torch.nn.Linear(16, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def run(n_states, disc, epsilon, episode_steps, lr, iters):

    # rewards and actions, made dynamically based on the number of states
    rew_a = np.zeros(n_states)
    rew_a[-1] = 1
    rew_b = np.zeros(n_states)
    rew_b[0] = 0.2
    # actions
    ac_a = np.arange(1, n_states+1)
    ac_a[-1] = ac_a[-2]  # last action is stay in last state
    ac_b = np.zeros(n_states, dtype=int)

    losses = []

    # init NN
    model = NN()  # S x A as input, Q(s, a) as output
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for _ in tqdm(range(iters)):

        # reset the environment
        state = 0

        preds = []
        targets = []

        # play episode
        for step in range(episode_steps):  # no terminal state, so just a fixed number of steps

            # choose action
            if np.random.random() < epsilon:
                action = np.random.choice([0, 1])
            else:
                q_0 = model(torch.tensor([state, 0], dtype=torch.float32))
                q_1 = model(torch.tensor([state, 1], dtype=torch.float32))
                action = np.argmax([q_0.detach().numpy(), q_1.detach().numpy()])

            # next state
            next_state = ac_a[state] if action == 0 else ac_b[state]
            # get reward
            reward = rew_a[state] if action == 0 else rew_b[state]

            # put in replay memory
            pred = model(torch.tensor([state, action], dtype=torch.float32))
            preds.append(pred)

            # target should not have gradient
            with torch.no_grad():
                q_next = model(torch.tensor([next_state, 0], dtype=torch.float32))
                q_next = max(q_next, model(torch.tensor([next_state, 1], dtype=torch.float32)))
                target = reward + disc * q_next
                # also put in replay memory
                targets.append(target)

            # update state and loop
            state = next_state

        # make preds and targets a tensor
        preds = torch.stack(preds)
        targets = torch.stack(targets)
        # sample 32 random indices
        idx = np.random.randint(0, len(preds), 32)
        # get the loss and backprop
        loss = loss_fn(preds[idx], targets[idx])
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # make final Q table
    Q_table = np.zeros((n_states, 2))
    for i in range(n_states):
        Q_table[i, 0] = model(torch.tensor([i, 0], dtype=torch.float32)).detach().numpy()
        Q_table[i, 1] = model(torch.tensor([i, 1], dtype=torch.float32)).detach().numpy()

    return Q_table, losses


def plot_policy_heatmap(Q_tab):
    plt.figure()
    plt.imshow(Q_tab, cmap='magma')
    plt.xticks([0, 1], ['A', 'B'])
    plt.xlabel('Action')
    plt.ylabel('State')
    plt.title('Q table')
    plt.colorbar()
    plt.show()


def plot_loss(losses):
    plt.plot(losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.show()


n_states = 10
disc = 0.9
epsilon = 0.4
episode_steps = 100
lr = 1e-3
iters = 4000

Q_table, losses = run(n_states, disc, epsilon, episode_steps, lr, iters)

plot_policy_heatmap(Q_table)
plot_loss(losses)
