import pygame
import time

from SnakeGame import SnakeGame
from Agent import Agent
from models.Action import Action

# controls how big one block is drawn (in pixels) on the game screen
BLOCK_SIZE_PIXELS = 50

# number of blocks (width, height) for the board
BOARD_SIZE_BLOCKS = (20, 10)

def main():
    # initialize pygame
    pygame.init()

    game = SnakeGame(BOARD_SIZE_BLOCKS, BLOCK_SIZE_PIXELS)

    # create the agent
    agent = Agent()

    # load a checkpoint (if one exists)
    agent.load("snake_dqn_model.pth")
    
    best_total_reward = 0

    for episode in range(100000):
        state = game.reset()
        total_reward = 0

        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game.close()
                    exit()

            action: Action = agent.act(state)
            next_state, reward, done = game.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            game.render()
            time.sleep(0.1)

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

        agent.replay()
        agent.decay_epsilon()
        game.clock.tick(100000)  # control the game speed
        
        if total_reward > best_total_reward:
            best_total_reward = total_reward
            print(f"New best total reward: Episode {episode + 1} with {total_reward}")
            agent.save("snake_dqn_model.pth")

        # save the model every 100 episodes
        if (episode + 1) % 100 == 0:
            agent.save("snake_dqn_model.pth")


if __name__ == "__main__":
    main()
