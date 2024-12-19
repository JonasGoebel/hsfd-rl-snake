import numpy as np
import pygame
import random

from models.Action import Action
from models.Colors import Colors


class SnakeGame:
    """Environment wrapper for the Snake game."""

    def __init__(self, board_size_blocks: tuple[int, int], block_size_pixels: int):
        self.board_size_blocks = board_size_blocks
        self.block_size_pixels = block_size_pixels
        self.is_display_initalized = False
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """Resets the game to the initial state."""
        self.x = round(self.board_size_blocks[0] // 2)
        self.y = round(self.board_size_blocks[1] // 2)
        self.dx, self.dy = 0, 0
        self.snake = [[self.x, self.y]]
        self.food = self._place_food()
        self.score = 0
        self.done = False
        self.step_counter = 0  # Track steps since last food eaten
        return self.get_state()

    def _place_food(self):
        food = None
        while food is None:
            food = [
                random.randrange(0, self.board_size_blocks[0]),
                random.randrange(0, self.board_size_blocks[1]),
            ]
            if food in self.snake:
                food = None
        return food

    def step(self, action: Action):
        """Takes an action and updates the game state."""
        if action == Action.LEFT:
            self.dx, self.dy = -1, 0
        elif action == Action.UP:
            self.dx, self.dy = 0, -1
        elif action == Action.RIGHT:
            self.dx, self.dy = 1, 0
        elif action == Action.DOWN:
            self.dx, self.dy = 0, 1

        self.x += self.dx
        self.y += self.dy
        new_head_position = [self.x, self.y]

        # Check for collisions
        if (
            self.x < 0
            or self.x >= self.board_size_blocks[0]
            or self.y < 0
            or self.y >= self.board_size_blocks[1]
            or new_head_position in self.snake
        ):
            self.done = True
            print("Death Reason: Collision with wall or itself")
            return self.get_state(), -10, self.done

        # Check if the snake eats the food
        reward = 10 if new_head_position == self.food else -0.1
        if new_head_position == self.food:
            self.food = self._place_food()
            self.step_counter = 0
        else:
            self.snake.pop(0)  # Remove the tail

        self.snake.append(new_head_position)

        # Timeout penalty
        self.step_counter += 1
        if self.step_counter > 200:
            self.done = True
            print("Timeout")
            return self.get_state(), -50, self.done

        return self.get_state(), reward, self.done

    def get_state(self):
        """Gets the current state representation."""
        state = [
            self.dx,
            self.dy,
            self.food[0] - self.x,
            self.food[1] - self.y,
            int(self.x <= 0 or [self.x - 1, self.y] in self.snake),  # left
            int(self.x >= self.board_size_blocks[0] - 1 or [self.x + 1, self.y] in self.snake),  # right
            int(self.y <= 0 or [self.x, self.y - 1] in self.snake),  # up
            int(self.y >= self.board_size_blocks[1] - 1 or [self.x, self.y + 1] in self.snake),  # down
            self.step_counter,  # Include step counter in the state
        ]
        return np.array(state, dtype=np.float32)

    def render(self):
        """Renders the game on the screen."""

        # inizialize screen on first render
        if not self.is_display_initalized:
            self.is_display_initalized = True
            board_size_pixels = (
                self.board_size_blocks[0] * self.block_size_pixels,
                self.board_size_blocks[1] * self.block_size_pixels,
            )
            self.display = pygame.display.set_mode(board_size_pixels)
        
        # background color
        self.display.fill(Colors.BLACK.value)

        # Draw the snake
        for segment in self.snake:
            pygame.draw.rect(
                self.display,
                Colors.GREEN.value,
                pygame.Rect(
                    segment[0] * self.block_size_pixels, # x part
                    segment[1] * self.block_size_pixels, # y part
                    self.block_size_pixels,
                    self.block_size_pixels,
                )
            )

        # Draw the food
        pygame.draw.rect(
            self.display,
            Colors.RED.value,
            pygame.Rect(
                self.food[0] * self.block_size_pixels,
                self.food[1] * self.block_size_pixels,
                self.block_size_pixels,
                self.block_size_pixels,
            ),
        )

        pygame.display.flip()
