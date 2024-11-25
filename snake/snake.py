import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import pygame
import os
import time

# Game and RL Parameters
WIDTH, HEIGHT = 600, 400
BLOCK_SIZE = 50
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
BATCH_SIZE = 64
MAX_MEMORY = 10000
STATE_SIZE = 7
ACTION_SIZE = 4

class DQN(nn.Module):
    """Deep Q-Network for learning the Q-values."""
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize pygame
pygame.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

class SnakeGameAI:
    """Environment wrapper for the Snake game."""
    def __init__(self):
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake Game AI")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """Resets the game to the initial state."""
        self.x, self.y = WIDTH // 2, HEIGHT // 2
        self.dx, self.dy = 0, 0
        self.snake = [[self.x, self.y]]
        self.food = self._place_food()
        self.score = 0
        self.done = False
        self.last_position = (self.x, self.y)  # Track the last position of the snake
        self.prev_last_position = (self.x, self.y)  # Track the second-to-last position
        self.step_counter = 0  # Step counter to track timeouts
        return self.get_state()

    def _place_food(self):
        """Places food at a random location."""
        return [
            round(random.randrange(0, WIDTH - BLOCK_SIZE) / BLOCK_SIZE) * BLOCK_SIZE,
            round(random.randrange(0, HEIGHT - BLOCK_SIZE) / BLOCK_SIZE) * BLOCK_SIZE
        ]

    def step(self, action):
        """Takes an action and updates the game state."""
        if action == 0: self.dx, self.dy = -BLOCK_SIZE, 0  # Left
        elif action == 1: self.dx, self.dy = BLOCK_SIZE, 0  # Right
        elif action == 2: self.dx, self.dy = 0, -BLOCK_SIZE  # Up
        elif action == 3: self.dx, self.dy = 0, BLOCK_SIZE  # Down

        self.x += self.dx
        self.y += self.dy
        new_head = [self.x, self.y]

        # Check if the snake has hit the wall or itself
        if self.x < 0 or self.x >= WIDTH or self.y < 0 or self.y >= HEIGHT or new_head in self.snake[:-1]:
            self.done = True
            return self.get_state(), -10, self.done

        reward = -0#.1  # Default reward for moving  

        # Check if the snake has eaten food
        if new_head == self.food:
            self.food = self._place_food()
            reward = 10
            self.score += 1
            self.step_counter = 0  # Reset step counter after eating food
        else:
            self.snake.pop(0)
            self.step_counter += 1  # Increment step counter

        # Timeout penalty if 1000 steps passed without eating food
        if self.step_counter > 1000:
            self.done = True
            reward = -10

        # Add the new head to the snake
        self.snake.append(new_head)

        # Apply a negative reward for moving back to the second-to-last position
        if new_head == self.prev_last_position:
            reward = -5
        # Update position tracking
        self.prev_last_position = self.last_position
        self.last_position = new_head

        return self.get_state(), reward, self.done



    def get_state(self):
        state = [
            self.dx, self.dy,
            self.food[0] - self.x, self.food[1] - self.y,
            int([self.x - BLOCK_SIZE, self.y] in self.snake),
            int([self.x + BLOCK_SIZE, self.y] in self.snake),
            int([self.x, self.y - BLOCK_SIZE] in self.snake)
        ]
        return np.array(state, dtype=np.float32)

    def render(self):
        """Renders the game on the screen."""
        self.display.fill(BLACK)

        # Draw snake
        for segment in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(segment[0], segment[1], BLOCK_SIZE, BLOCK_SIZE))

        # Draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))

        # Display score
        font = pygame.font.SysFont(None, 35)
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(score_text, (10, 10))

        pygame.display.flip()

    def close(self):
        """Closes the game window."""
        pygame.quit()






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

    def act(self, state):
        """Selects an action based on epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(ACTION_SIZE)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            actions = self.model(state)
        return torch.argmax(actions).item()

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
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)

    def load(self, filename):
        """Loads the model and optimizer state from a file."""
        if os.path.exists(filename):
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"Model loaded from {filename}")
        else:
            print(f"No checkpoint found at {filename}")

if __name__ == "__main__":
    game = SnakeGameAI()
    
    # Create the agent
    agent = Agent()

    # Optionally load a checkpoint if exists
    agent.load("snake_dqn_model.pth")

    for episode in range(100000):
        state = game.reset()
        total_reward = 0

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game.close()
                    exit()

            action = agent.act(state)
            next_state, reward, done = game.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            game.render()
            time.sleep(0.1)

            if done:
                print(f"Episode {episode + 1}: Total Reward: {total_reward}")
                break

        agent.replay()
        agent.decay_epsilon()
        game.clock.tick(100000)  # Control the game speed

        # Save the model every 100 episodes
        if (episode + 1) % 100 == 0:
            agent.save("snake_dqn_model.pth")
