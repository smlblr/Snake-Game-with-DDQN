import sys

import numpy as np
import pygame


class environment_init:

    def __init__(self, window_width=704, window_height=704, board_size_y=32):
        assert (board_size_y * (window_width / window_height)) % 1 == 0, "Window width must be compatible number."
        assert window_width % board_size_y == 0, "Window width must be divisible by the board size."
        assert window_height % board_size_y == 0, "Window height must be divisible by the board size."
        pygame.init()
        self.window_width = window_width
        self.window_height = window_height
        self.display = pygame.display.set_mode((self.window_width, self.window_height))
        self.font = pygame.font.Font('freesansbold.ttf', 16)
        pygame.display.set_caption('PUNGI')
        self.food_spawn = True
        self.score = 0
        self.highestscore = 0
        self.restart = False
        self.board_size_y = board_size_y
        self.board_size_x = int(self.board_size_y * (self.window_width / self.window_height))
        self.block_size = int(min(self.window_width, self.window_height) / min(self.board_size_x, self.board_size_y))

    def show_score(self):
        score_render = self.font.render("Score : " + str(self.score), True, (255, 255, 255))
        self.display.blit(score_render, (0, 0))

    def show_highestscore(self):
        if self.score > self.highestscore:
            self.highestscore = self.score
        highestscore_render = self.font.render("Highest Score : " + str(self.highestscore), True, (255, 255, 255))
        self.display.blit(highestscore_render, (100, 0))

    def wall(self):
        for wall_location_x in range(0, self.window_width, self.block_size):
            wall_rect = pygame.Rect(wall_location_x, 0, playground_init.block_size, playground_init.block_size)
            pygame.draw.rect(playground_init.display, (255, 0, 0), wall_rect)
            wall_rect = pygame.Rect(wall_location_x, self.block_size * (self.board_size_y - 1),
                                    playground_init.block_size, playground_init.block_size)
            pygame.draw.rect(playground_init.display, (255, 0, 0), wall_rect)
        for wall_location_y in range(0, self.window_height, self.block_size):
            wall_rect = pygame.Rect(0, wall_location_y, playground_init.block_size, playground_init.block_size)
            pygame.draw.rect(playground_init.display, (255, 0, 0), wall_rect)
            wall_rect = pygame.Rect(self.block_size * (self.board_size_x - 1), wall_location_y, playground_init.block_size, playground_init.block_size)
            pygame.draw.rect(playground_init.display, (255, 0, 0), wall_rect)

    # drawing grid lines
    def grid_lines(self):
        for x in range(0, self.window_width, self.block_size):  # drawing vertical lines
            pygame.draw.line(self.display, (40, 40, 40), (x, 0), (x, self.window_height))
        for y in range(0, self.window_height, self.block_size):  # drawing horizontal lines
            pygame.draw.line(self.display, (40, 40, 40), (0, y), (self.window_width, y))

    def update_window(self):
        pygame.display.update()


class Snake(object):

    def __init__(self):
        self.snake_head_x = playground_init.window_width / 2 - playground_init.block_size
        self.snake_head_y = playground_init.window_height / 2 - playground_init.block_size
        self.change_x = 0
        self.change_y = 0
        self.tailsX = [playground_init.window_width / 2 - playground_init.block_size]
        self.tailsY = [playground_init.window_height / 2 - playground_init.block_size]
        # self.tail_length = len(self.tailsX)
        self.counter = 0

    def move(self):
        for i in range(len(self.tailsX) - 1, 0, -1):
            self.tailsX[i] = self.tailsX[i - 1]
            self.tailsY[i] = self.tailsY[i - 1]
        self.tailsX[0] = self.snake_head_x
        self.tailsY[0] = self.snake_head_y
        self.snake_head_x += snake.change_x
        self.snake_head_y += snake.change_y

    def snake_head_location(self):
        snake_head_rect = pygame.Rect(self.snake_head_x, self.snake_head_y, playground_init.block_size,
                                      playground_init.block_size)
        pygame.draw.rect(playground_init.display, (0, 255, 0), snake_head_rect)

    # Adding tail if the snake eats food
    def add_tail(self):
        self.tailsX.append(self.tailsX[len(self.tailsX) - 1])
        self.tailsY.append(self.tailsY[len(self.tailsY) - 1])

    # Arranging snake's tail location
    def snake_tail_location(self):
        for i in range(0, len(self.tailsX)):
            if self.counter == 0:
                snake_tail_rect = pygame.Rect(self.tailsX[i] + playground_init.block_size, self.tailsY[i],
                                              playground_init.block_size,
                                              playground_init.block_size)
                pygame.draw.rect(playground_init.display, (0, 125, 0), snake_tail_rect)
            else:
                snake_tail_rect = pygame.Rect(self.tailsX[i], self.tailsY[i],
                                              playground_init.block_size, playground_init.block_size)
                pygame.draw.rect(playground_init.display, (0, 125, 0), snake_tail_rect)


class Food(object):

    def __init__(self):
        self.food_state = True
        self.food_location()

    def eat(self):
        if snake.snake_head_x == self.food_x and snake.snake_head_y == self.food_y:
            playground_init.score += 1
            # snake.tail_length = len(snake.tailsX)
            snake.add_tail()
            self.food_state = True
            self.food_location()
            self.food_spawn()

    # Arranging food spawn location
    def food_location(self):
        if self.food_state:
            self.food_x = np.random.choice(
                np.arange(start=playground_init.block_size,
                          stop=playground_init.window_width - playground_init.block_size,
                          step=playground_init.block_size))
            self.food_y = np.random.choice(
                np.arange(start=playground_init.block_size,
                          stop=playground_init.window_height - playground_init.block_size,
                          step=playground_init.block_size))

        # Arranging food location not to spawn on snake
        for i in range(0, len(snake.tailsX) - 1):
            if self.food_x == snake.tailsX[i] and self.food_y == snake.tailsY[i]:
                self.food_x = np.random.choice(
                    np.arange(start=playground_init.block_size,
                              stop=playground_init.window_width - playground_init.block_size,
                              step=playground_init.block_size))
                self.food_y = np.random.choice(
                    np.arange(start=playground_init.block_size,
                              stop=playground_init.window_height - playground_init.block_size,
                              step=playground_init.block_size))
                i = 0
        self.food_state = False
        return self.food_x, self.food_y

    def food_spawn(self):
        food_rect = pygame.Rect(self.food_x, self.food_y, playground_init.block_size, playground_init.block_size)
        pygame.draw.rect(playground_init.display, (0, 0, 255), food_rect)


class Run:

    def __init__(self):
        self.key_pressed = False
        self.run_state = True
        self.t_o_x = None
        self.t_o_y = None
        self.h_o_x = None
        self.h_o_y = None

    def buttonPress(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.run_state = False
                self.exitgame()
            # Check users press any keys
            elif event.type == pygame.KEYDOWN and self.key_pressed == False:
                self.take_action(event)

    def take_action(self, event):
        if (event.key == pygame.K_LEFT or event.key == pygame.K_a) and snake.change_x != playground_init.block_size:
            snake.change_x = -playground_init.block_size
            snake.change_y = 0
            snake.counter += 1
            self.key_pressed = True
        elif (event.key == pygame.K_RIGHT or event.key == pygame.K_d) and \
                snake.change_x != -playground_init.block_size and snake.counter != 0:
            snake.change_x = playground_init.block_size
            snake.change_y = 0
            snake.counter += 1
            self.key_pressed = True
        elif (event.key == pygame.K_UP or event.key == pygame.K_w) and snake.change_y != playground_init.block_size:
            snake.change_y = -playground_init.block_size
            snake.change_x = 0
            snake.counter += 1
            self.key_pressed = True
        elif (event.key == pygame.K_DOWN or event.key == pygame.K_s) and snake.change_y != -playground_init.block_size:
            snake.change_y = playground_init.block_size
            snake.change_x = 0
            snake.counter += 1
            self.key_pressed = True
        elif event.key == pygame.K_ESCAPE:
            self.exitgame()
        elif event.key == pygame.K_SPACE:
            pause_press = True
            self.pause_game(pause_press)

    # Check if the snake has hit the wall
    def hit_wall(self):
        # Rearranging snake's head and tail if the snake has hit the wall
        if snake.snake_head_x == playground_init.block_size or snake.snake_head_y == playground_init.block_size \
                or snake.snake_head_x == playground_init.block_size * (playground_init.board_size_x - 2) \
                or snake.snake_head_y == playground_init.block_size * (playground_init.board_size_y - 2):
            self.t_o_x = snake.tailsX[len(snake.tailsX) - 1]
            self.t_o_y = snake.tailsY[len(snake.tailsX) - 1]
            self.h_o_x = snake.snake_head_x
            self.h_o_y = snake.snake_head_y

        # Restarting game if the snake has hit the wall
        if snake.snake_head_x == 0 or snake.snake_head_y == 0 \
                or snake.snake_head_x == playground_init.block_size * (playground_init.board_size_x - 1) \
                or snake.snake_head_y == playground_init.block_size * (playground_init.board_size_y - 1):
            head_before_die_rect = pygame.Rect(self.h_o_x, self.h_o_y, playground_init.block_size,
                                               playground_init.block_size)
            pygame.draw.rect(playground_init.display, (0, 255, 0), head_before_die_rect)
            tail_before_die_rect = pygame.Rect(self.t_o_x, self.t_o_y, playground_init.block_size,
                                               playground_init.block_size)
            pygame.draw.rect(playground_init.display, (0, 125, 0), tail_before_die_rect)
            self.restart()

    # Check if the snake has eaten itself
    def hit_itself(self):
        if len(snake.tailsX) > 1 or len(snake.tailsY) > 1:
            for i in range(1, len(snake.tailsX) - 1):
                for j in range(1, len(snake.tailsY) - 1):
                    if snake.snake_head_x == snake.tailsX[i] and snake.snake_head_y == snake.tailsY[i]:
                        self.restart()
                        break

    def pause_game(self, pause_press):
        while pause_press:
            for eventspace in pygame.event.get():
                if eventspace.type == pygame.KEYDOWN:
                    if eventspace.key == pygame.K_SPACE:
                        pause_press = False

    def restart(self):
        snake.change_x = 0
        snake.change_y = 0
        playground_init.score = 0
        snake.counter = 0
        snake.snake_head_x = playground_init.window_width / 2 - playground_init.block_size
        snake.snake_head_y = playground_init.window_width / 2 - playground_init.block_size
        snake.tailsX = [playground_init.window_width / 2 - playground_init.block_size]
        snake.tailsY = [playground_init.window_height / 2 - playground_init.block_size]
        food.food_state = True
        food.food_location()
        food.food_spawn()

    def exitgame(self):
        pygame.quit()
        sys.exit()

    def running(self):
        while self.run_state:
            self.buttonPress()

            playground_init.display.fill((0, 0, 0))
            playground_init.grid_lines()

            snake.move()
            food.eat()

            snake.snake_head_location()
            snake.snake_tail_location()

            food.food_spawn()
            self.hit_wall()
            self.hit_itself()

            playground_init.wall()

            playground_init.show_score()
            playground_init.show_highestscore()
            playground_init.update_window()
            self.key_pressed = False

            pygame.time.wait(75)


playground_init = environment_init(768, 768, 32)
snake = Snake()
food = Food()
run = Run()

if __name__ == '__main__':
    run.running()
