import numpy as np
import random

# Parameters
grid_size = 5
episodes = 1000
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1
epsilon_decay = 0.995
learning_rate = 0.1
gamma = 0.95  # Discount factor

# Q-learning Table
q_table = np.zeros((grid_size, grid_size, 4))  # Actions: Up, Down, Left, Right

# Environment (5x5 grid)
traffic_signals = [(1, 2), (2, 3)]
obstacles = [(3, 1), (1, 4)]
goal = (4, 4)  # Destination

# Helper functions
def get_reward(state):
    if state == goal:
        return 10
    elif state in obstacles:
        return -10
    elif state in traffic_signals:
        return -1
    return -0.1

def get_next_state(state, action):
    x, y = state
    if action == 0:  # Up
        x = max(x - 1, 0)
    elif action == 1:  # Down
        x = min(x + 1, grid_size - 1)
    elif action == 2:  # Left
        y = max(y - 1, 0)
    elif action == 3:  # Right
        y = min(y + 1, grid_size - 1)
    return (x, y)

# Q-learning training loop
for episode in range(episodes):
    state = (0, 0)
    total_reward = 0

    for step in range(100):  # Max steps per episode
        if random.uniform(0, 1) < epsilon:
            action = random.choice([0, 1, 2, 3])  # Explore
        else:
            action = np.argmax(q_table[state[0], state[1], :])  # Exploit

        next_state = get_next_state(state, action)
        reward = get_reward(next_state)

        # Q-learning update
        best_next_action = np.argmax(q_table[next_state[0], next_state[1], :])
        q_table[state[0], state[1], action] = q_table[state[0], state[1], action] + \
            learning_rate * (reward + gamma * q_table[next_state[0], next_state[1], best_next_action] - q_table[state[0], state[1], action])

        state = next_state
        total_reward += reward

        if state == goal:
            break

    # Decay epsilon for exploration-exploitation balance
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

print("Training complete.")
print("Trained Q-table:\n", q_table)
