from collections import deque
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

from DQN import DQN
from models.Action import Action

LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
BATCH_SIZE = 64
MAX_MEMORY = 10000
STATE_SIZE = 7
ACTION_SIZE = 4


class Agent:
    """Reinforcement learning agent using DQN."""

    def __init__(self, model=None):
        self.memory = deque(maxlen=MAX_MEMORY)
        self.epsilon = 1.0
        self.model = model if model else DQN(STATE_SIZE, ACTION_SIZE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        """Stores a transition in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state) -> Action:
        """Selects an action based on epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(ACTION_SIZE)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            actions = self.model(state)
        return Action(torch.argmax(actions).item())

    def replay(self):
        """Trains the model on a batch of experiences."""
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Ensure all data are numpy arrays of the correct shape
        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Forward pass for Q-values
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = self.model(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * GAMMA * next_q_values

        # Compute loss and optimize
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        """Reduces epsilon for exploration-exploitation tradeoff."""
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY

    def save(self, filename):
        """Saves the model and optimizer state to a file."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            filename,
        )

    def load(self, filename):
        """Loads the model and optimizer state from a file."""
        if os.path.exists(filename):
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epsilon = checkpoint["epsilon"]
            print(f"Model loaded from {filename}")
        else:
            print(f"No checkpoint found at {filename}")
