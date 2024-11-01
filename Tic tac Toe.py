import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Tic Tac Toe game class
class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]
        self.winner = None

    def reset(self):
        self.board = [' ' for _ in range(9)]
        self.winner = None

    def available_moves(self):
        return [i for i, x in enumerate(self.board) if x == ' ']

    def make_move(self, square, player):
        if self.board[square] == ' ':
            self.board[square] = player
            if self.check_winner(player):
                self.winner = player
            return True
        return False

    def check_winner(self, player):
        win_patterns = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                        (0, 3, 6), (1, 4, 7), (2, 5, 8),
                        (0, 4, 8), (2, 4, 6)]
        return any(self.board[a] == self.board[b] == self.board[c] == player for a, b, c in win_patterns)


# Q-learning agent for Tic Tac Toe
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0)

    def choose_action(self, state, available_moves):
        if random.random() < self.epsilon:
            return random.choice(available_moves)
        q_values = [self.get_q(state, a) for a in available_moves]
        max_q = max(q_values)
        return available_moves[q_values.index(max_q)]

    def update_q(self, state, action, reward, next_state, done):
        old_q = self.q_table.get((state, action), 0)
        max_next_q = max([self.get_q(next_state, a) for a in range(9)], default=0)
        if done:
            self.q_table[(state, action)] = old_q + self.alpha * (reward - old_q)
        else:
            self.q_table[(state, action)] = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)


# Training function
def train(agent, game, episodes=1000):
    for episode in range(episodes):
        game.reset()
        state = tuple(game.board)
        done = False
        while not done:
            action = agent.choose_action(state, game.available_moves())
            game.make_move(action, 'X')
            reward = 1 if game.winner == 'X' else -1 if game.winner == 'O' else 0
            next_state = tuple(game.board)
            done = game.winner is not None or ' ' not in game.board
            agent.update_q(state, action, reward, next_state, done)
            state = next_state


# Convert game state to numeric format for clustering
def state_to_numeric(state):
    return [1 if s == 'X' else -1 if s == 'O' else 0 for s in state]


# Cluster states using K-means and visualize the results
def cluster_and_visualize(q_table):
    states = [state for state, action in q_table.keys()]
    unique_states_numeric = np.array([state_to_numeric(state) for state in set(states)])

    # Reduce dimensions with PCA for visualization
    pca = PCA(n_components=2)
    reduced_states = pca.fit_transform(unique_states_numeric)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=0)
    labels = kmeans.fit_predict(reduced_states)

    # Plotting the clusters
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_states[:, 0], reduced_states[:, 1], c=labels, cmap='viridis', s=50)
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Clustering of Tic Tac Toe Game States")
    plt.show()

    return labels


# Initialize game and agent
game = TicTacToe()
agent = QLearningAgent()

# Train agent
train(agent, game, episodes=10000)

# Cluster and visualize state patterns
labels = cluster_and_visualize(agent.q_table)
print("Clustering results for each state:\n", labels)
