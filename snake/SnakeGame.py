import numpy as np
import pygame
import random

from models.Action import Action
from models.Colors import Colors


class SnakeGame:
    """Environment wrapper for the Snake game."""

    def __init__(
        self, board_size_blocks: set[int, int], block_size_pixels: int
    ):
        # board size in blocks
        self.board_size_blocks = board_size_blocks
        
        # size per block in pixels (for window size)
        self.block_size_pixels = block_size_pixels
        
        # board size in pixels (window size)
        board_size_pixels = (
            board_size_blocks[0] * block_size_pixels,
            board_size_blocks[1] * block_size_pixels,
        )

        self.display = pygame.display.set_mode(board_size_pixels)
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """Resets the game to the initial state."""

        # place snake at starting position (center of map)
        self.x = round(self.board_size_blocks[0] // 2)
        self.y = round(self.board_size_blocks[1] // 2)

        self.dx, self.dy = 0, 0
        self.snake = [[self.x, self.y]]
        self.food = self._place_food()
        self.score = 0
        self.done = False
        self.last_position = (
            self.x,
            self.y,
        )  # track the last position of the snake
        self.prev_last_position = (
            self.x,
            self.y,
        )  # track the second-to-last position
        self.step_counter = 0  # step counter to track timeouts
        return self.get_state()

    def _place_food(self):
        food = None
        while food is None:
            food = [
                random.randrange(0, self.board_size_blocks[0]),
                random.randrange(0, self.board_size_blocks[1]),
            ]

            # check if food position is on the snake
            if food in self.snake:
                food = None  # reattempt placing food if it spawns on the snake

        return food

    def step(self, action: Action):
        """Takes an action and updates the game state."""
        if action == Action.LEFT:
            self.dx = -1
            self.dy = 0
        elif action == Action.UP:
            self.dx = 0
            self.dy = 1
        elif action == Action.RIGHT:
            self.dx = 1
            self.dy = 0
        elif action == Action.DOWN:
            self.dx = 0
            self.dy = -1

        # to the step
        self.x += self.dx
        self.y += self.dy
        new_head_position = [self.x, self.y]

        # check if the snake has hit the wall or itself
        if (
            # snake hit wall
            self.x < 0
            or self.x >= self.board_size_blocks[0]
            or self.y < 0
            or self.y >= self.board_size_blocks[1]
            # snake hit itself
            or new_head_position in self.snake[:-1]
        ):
            # cancel game with penalty
            self.done = True
            return self.get_state(), -10, self.done

        reward = -0  # .1  # default reward for moving

        # check if the snake has eaten food
        if new_head_position == self.food:
            self.food = self._place_food()
            reward = 10
            self.score += 1
            self.step_counter = 0  # reset step counter after eating food
        else:
            self.snake.pop(0)
            self.step_counter += 1  # increment step counter

        # timeout penalty if 1000 steps passed without eating food
        if self.step_counter > 1000:
            self.done = True
            reward = -10

        # add the new head to the snake
        self.snake.append(new_head_position)

        # apply a negative reward for moving back to the second-to-last position
        if new_head_position == self.prev_last_position:
            reward = -5
        # update position tracking
        self.prev_last_position = self.last_position
        self.last_position = new_head_position

        return self.get_state(), reward, self.done

    def get_state(self):
        state = [
            # snake position
            self.dx,
            self.dy,
            # position in relation to food
            self.food[0] - self.x,
            self.food[1] - self.y,

            # 0/1 (boolean converted to int)
            int([self.x - 1, self.y] in self.snake),
            int([self.x + 1, self.y] in self.snake),
            int([self.x, self.y - 1] in self.snake),
        ]
        return np.array(state, dtype=np.float32)

    def __draw_block(self, x: int, y: int, color):
        rect_x = x * self.block_size_pixels
        rect_y = y * self.block_size_pixels
        pygame.draw.rect(
            self.display,
            color,
            pygame.Rect(
                rect_x, rect_y, self.block_size_pixels, self.block_size_pixels
            ),
        )

    def render(self):
        """Renders the game on the screen."""
        self.display.fill(Colors.BLACK.value)

        # draw snake parts
        for segment in self.snake:
            self.__draw_block(segment[0], segment[1], Colors.GREEN.value)

        # draw food
        self.__draw_block(self.food[0], self.food[1], Colors.RED.value)

        # draw score
        font = pygame.font.SysFont(None, 35)
        score_text = font.render(f"Score: {self.score}", True, Colors.WHITE.value)
        self.display.blit(score_text, (10, 10))

        pygame.display.flip()

    def close(self):
        """Closes the game window."""
        pygame.quit()
