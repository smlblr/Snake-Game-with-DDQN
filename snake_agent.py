# Yılan duvarın içine geçiyor kendi üzerinde yem oluşuyor .
# batch_size 500 dene
# her ilk hareket random
# Highscore grafiği
# Her Skorda kaç adım atmış pie grafiği ya da heatmap grafiği dot grafiğin alternatifi
# Keras plot model loss (https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/)
# Keras modeli blok diagram halinde çizdirme (https://keras.io/api/utils/model_plotting_utils/)


# !pip uninstall tensorflow
# !pip install tensorflow-gpu
# !pip install pygame
# !pip install matplotlib==3.3.4
# !pip install --upgrade pip
# !pip install opencv-contrib-python
# !pip install opencv-python-headless
# !pip install seaborn

import sys
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ['SDL_VIDEODRIVER'] = 'dummy'

import pygame
import numpy as np
import cv2
from DDQN2 import DDQNAgent

from GraphFunc import drawing_graph
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# sns.set()
# plt.style.use('seaborn-colorblind')

np.set_printoptions(threshold=sys.maxsize)


class environment_init:

    def __init__(self, env_pixel, board_size, resize_input_dims, take_graph, graph_per_episode, pers_of_kernel,
                 number_of_food, food_decay_step, input_ch_num=4, gui=True):
        assert (board_size[1] * (env_pixel[0] / env_pixel[1])) % 1 == 0, "Window width must be compatible number."
        assert env_pixel[0] % board_size[0] == 0, "Window width must be divisible by the board size."
        assert env_pixel[1] % board_size[1] == 0, "Window height must be divisible by the board size."
        pygame.init()
        self.take_graph = take_graph
        self.graph_per_episode = graph_per_episode
        self.env_pixel = env_pixel
        self.number_of_food = number_of_food
        self.old_number_of_food = number_of_food+1
        self.food_decay_step = food_decay_step
        self.pers_of_kernel = pers_of_kernel
        self.input_ch_num = input_ch_num
        self.font = pygame.font.Font('freesansbold.ttf', 14)
        pygame.display.set_caption('PUNGI')
        self.restart = False
        self.board_size = board_size
        self.block_size = int(min(self.env_pixel[0], self.env_pixel[1]) / min(self.board_size[0], self.board_size[1]))
        self.do_resize = True
        self.gui = gui
        self.state_flash_x = 0
        self.state_flash_y = 0
        self.st_flash = None
        self.new_st_flash = None
        self.st_flash_not_rot_surface = None
        self.new_st_flash_not_rot_surface = None
        self.st_flash_rot_surface = None
        self.new_st_flash_rot_surface = None
        self.st_flash_not_rot_gray_surf = None
        self.new_st_flash_not_rot_gray_surf = None
        self.pic_counter = 0
        self.pers_fix = [0, 0]
        self.resize_input_dims = resize_input_dims
        if resize_input_dims[0] == pers_of_kernel[0] * self.block_size \
                and resize_input_dims[1] == pers_of_kernel[1] * self.block_size:
            self.do_resize = False

        if self.gui:
            self.env_window = pygame.Surface((self.env_pixel[0], self.env_pixel[1]))
            self.gui_window = pygame.display.set_mode((64 + self.env_pixel[0] + 64 + self.pers_of_kernel[0] *
                                                       self.block_size + 64 + self.pers_of_kernel[0] * self.block_size
                                                       + 64, 64 + self.env_pixel[1] + 200))
        else:
            self.env_window = pygame.display.set_mode((self.env_pixel[0], self.env_pixel[1]))

    def gui_blit(self):
        self.gui_window.fill((75, 75, 75))
        self.gui_window.blit(self.env_window, (64, 64))
        if ddqn_agent.t == 0:
            self.gui_window.blit(self.st_flash_not_rot_surface,
                                 (64 + self.env_pixel[0] + 64,
                                  64 + (self.env_pixel[1] - self.pers_of_kernel[1] * self.block_size) / 2))
        else:
            self.gui_window.blit(self.new_st_flash_not_rot_surface,
                                 (64 + self.env_pixel[0] + 64,
                                  64 + (self.env_pixel[1] - self.pers_of_kernel[1] * self.block_size) / 2))

        score_render = self.font.render("Score: " + str(run.score), True, (255, 255, 255))
        self.gui_window.blit(score_render, (10, 2))

        high_score_render = self.font.render("Highest Score: " + str(run.highestscore), True, (255, 255, 255))
        self.gui_window.blit(high_score_render,
                             (64 + self.env_pixel[0] + 64 + self.pers_of_kernel[0] * self.block_size +
                              64 + self.pers_of_kernel[0] * self.block_size - 95, 2))

        episode_render = self.font.render("Episode Number: " + str(run.episode_number), True, (255, 255, 255))
        self.gui_window.blit(episode_render, (64, 64 + self.env_pixel[1] + 18))

        step_render = self.font.render("Step Number: " + str(ddqn_agent.t), True, (255, 255, 255))
        self.gui_window.blit(step_render, (64, 64 + self.env_pixel[1] + 18 + 30))

        average_render = self.font.render("Average Score: " + str(run.scores_pie_graph[-10:].mean()), True,
                                          (255, 255, 255))
        self.gui_window.blit(average_render, (64, 64 + self.env_pixel[1] + 18 + 60))

        epsilon_render = self.font.render("Epsilon: {:2.4f} ".format(ddqn_agent.epsilon), True, (255, 255, 255))
        self.gui_window.blit(epsilon_render, (64, 64 + self.env_pixel[1] + 18 + 90))

        remain_render = self.font.render("Max Step: " + str(run.maxIdleStep), True, (255, 255, 255))
        self.gui_window.blit(remain_render, (64, 64 + self.env_pixel[1] + 18 + 120))

        food_number = self.font.render("Food Number: " + str(self.number_of_food), True, (255, 255, 255))
        self.gui_window.blit(food_number, (64, 64 + self.env_pixel[1] + 18 + 150))

        env_render = self.font.render("Env.", True, (255, 255, 255))
        self.gui_window.blit(env_render, (64, 32 + 2))

        pers_render = self.font.render("Pers.", True, (255, 255, 255))
        self.gui_window.blit(pers_render, (64 + self.env_pixel[0] + 64, 32 + 2))

    def set_highestscore(self):
        if run.score > run.highestscore:
            run.highestscore = run.score
        if self.number_of_food == run.highestscore_counter:
            run.highestscore = 0
            run.highestscore_counter -= 1

        # highestscore_render = self.font.render("Highest Score : " + str(run.highestscore), True, (255, 255, 255))
        # self.env_window.blit(highestscore_render, (300, 0))

    def wall(self):
        for wall_location_x in range(0, self.env_pixel[0], self.block_size):
            pygame.draw.rect(self.env_window, (255, 0, 0), (wall_location_x, 0, self.block_size, self.block_size))

            pygame.draw.rect(self.env_window, (255, 0, 0), (wall_location_x, self.block_size * (self.board_size[1] - 1),
                                                            self.block_size, self.block_size))

        for wall_location_y in range(0, self.env_pixel[1], self.block_size):
            pygame.draw.rect(self.env_window, (255, 0, 0), (0, wall_location_y, self.block_size, self.block_size))

            pygame.draw.rect(self.env_window, (255, 0, 0), (self.block_size * (self.board_size[0] - 1), wall_location_y,
                                                            self.block_size, self.block_size))

    # drawing grid lines
    def grid_lines(self):
        for x in range(0, self.env_pixel[0], self.block_size):  # drawing vertical lines
            pygame.draw.line(self.env_window, (40, 40, 40), (x, 0), (x, self.env_pixel[1]))
        for y in range(0, self.env_pixel[1], self.block_size):  # drawing horizontal lines
            pygame.draw.line(self.env_window, (40, 40, 40), (0, y), (self.env_pixel[0], y))

    def update_window(self):
        pygame.display.update()

    def perspective_not_rotated_grayscale(self):
        if ddqn_agent.t == 0:
            # cv2.imwrite("resimler/zresim0_not_rotated_pers_gray.png", self.st_flash)
            if self.gui:
                self.st_flash_not_rot_gray_surf = pygame.surfarray.make_surface(self.st_flash.swapaxes(0, 1))
                self.gui_window.blit(self.st_flash_not_rot_gray_surf,
                                     (64 + self.env_pixel[0] + 64 + self.pers_of_kernel[0] * self.block_size + 64,
                                      64 + (self.env_pixel[1] - self.pers_of_kernel[1] * self.block_size) / 2))
        else:
            # cv2.imwrite("resimler/zresim" + str(self.pic_counter + 1) + "_not_rotated_pers_gray.png", self.new_st_flash)
            self.new_st_flash_not_rot_gray_surf = pygame.Surface((self.pers_of_kernel[0] * self.block_size,
                                                                  self.pers_of_kernel[1] * self.block_size))
            if self.gui:
                self.new_st_flash_not_rot_gray_surf = pygame.surfarray.make_surface(self.new_st_flash.swapaxes(0, 1))
                self.gui_window.blit(self.new_st_flash_not_rot_gray_surf,
                                     (64 + self.env_pixel[0] + 64 + self.pers_of_kernel[0] * self.block_size + 64,
                                      64 + (self.env_pixel[1] - self.pers_of_kernel[1] * self.block_size) / 2))

    def perspective_rotate_against_snake(self):
        def perspective_pic_rotate(st_flash_not_rot):
            if snake.change_x < 0:  # sol
                st_flash_rot = cv2.rotate(st_flash_not_rot, cv2.ROTATE_90_CLOCKWISE)
            elif snake.change_x > 0:  # sağ
                st_flash_rot = cv2.rotate(st_flash_not_rot, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif snake.change_y < 0:  # yukarı
                st_flash_rot = st_flash_not_rot
            else:  # aşağı
                st_flash_rot = cv2.rotate(st_flash_not_rot, cv2.ROTATE_180)
            return st_flash_rot

        if ddqn_agent.t == 0:
            self.st_flash = perspective_pic_rotate(self.st_flash)
            # cv2.imwrite("resimler/zresim0_rotated_pers.png", self.st_flash)
            if self.gui:
                self.st_flash_rot_surface = pygame.surfarray.make_surface(self.st_flash.swapaxes(0, 1))
                self.gui_window.blit(self.st_flash_rot_surface,
                                     (64 + self.env_pixel[0] + 64 + self.pers_of_kernel[0] * self.block_size + 64,
                                      64 + (self.env_pixel[1] - self.pers_of_kernel[1] * self.block_size) / 2))
        else:
            self.new_st_flash = perspective_pic_rotate(self.new_st_flash)
            # cv2.imwrite("resimler/zresim" + str(self.pic_counter + 1) + "_rotated_pers.png", self.new_st_flash)
            if self.gui:
                self.new_st_flash_rot_surface = pygame.surfarray.make_surface(self.new_st_flash.swapaxes(0, 1))
                self.gui_window.blit(self.new_st_flash_rot_surface,
                                     (64 + self.env_pixel[0] + 64 + self.pers_of_kernel[0] * self.block_size + 64,
                                      64 + (self.env_pixel[1] - self.pers_of_kernel[1] * self.block_size) / 2))

    def perspective_wall_prevent_overflow(self):
        if snake.snake_head[0] < self.env_pixel[0] / 2:
            self.pers_fix[0] = (self.pers_of_kernel[0] * self.block_size) / 2 - snake.snake_head[0]
        elif snake.snake_head[0] > self.env_pixel[0] / 2:
            self.pers_fix[0] = self.env_pixel[0] - (self.pers_of_kernel[0] * self.block_size) / 2 - snake.snake_head[0]
        else:
            self.pers_fix[0] = 0
        if snake.snake_head[1] < self.env_pixel[1] / 2:
            self.pers_fix[1] = (self.pers_of_kernel[1] * self.block_size) / 2 - snake.snake_head[1]
        elif snake.snake_head[1] > self.env_pixel[1] / 2:
            self.pers_fix[1] = self.env_pixel[1] - (self.pers_of_kernel[1] * self.block_size) / 2 - snake.snake_head[1]
        else:
            self.pers_fix[1] = 0

    def first_screenshoot(self):

        # self.perspective_wall_prevent_overflow()

        rect = pygame.Rect(-(snake.snake_head[0] - (self.pers_of_kernel[0] * self.block_size) / 2 + self.pers_fix[0]),
                           -(snake.snake_head[1] - (self.pers_of_kernel[1] * self.block_size) / 2 + self.pers_fix[1]),
                           self.pers_of_kernel[0] * self.block_size, self.pers_of_kernel[1] * self.block_size)
        self.st_flash_not_rot_surface = pygame.Surface((self.pers_of_kernel[0] * self.block_size,
                                                        self.pers_of_kernel[1] * self.block_size))
        pygame.Surface.blit(self.st_flash_not_rot_surface, self.env_window, rect)
        # pygame.image.save(self.env_window, "resimler/0env.png")
        # pygame.image.save(self.st_flash_not_rot_surface, "resimler/0not_rotated.png")
        self.st_flash = pygame.surfarray.array3d(self.st_flash_not_rot_surface)
        # self.st_flash = self.st_flash.swapaxes(0, 1)
        # cv2.imwrite("resimler/zresim0_not_rotated_pers.png", self.st_flash)

        # environment_resim = pygame.surfarray.array3d(self.env_window)
        # environment_resim = environment_resim.swapaxes(0, 1)
        # cv2.imwrite("resimler/zresim0_not_rotated_env.png", environment_resim)

        if self.gui:
            self.gui_blit()

        # # Perspektif resminin yılan hep yukarı doğru görünecek şekilde düzenlenmesi
        # self.perspective_rotate_against_snake()

        self.st_flash = cv2.cvtColor(self.st_flash, cv2.COLOR_BGR2GRAY).astype(float)
        # self.st_flash[self.st_flash == 159] = 116  # yem
        # self.st_flash[self.st_flash == 29] = 127  # duvar
        # self.st_flash[self.st_flash == 150] = 63  # yılan kafası
        # self.st_flash[self.st_flash == 73] = 127  # yılan kuyruğu

        # self.perspective_not_rotated_grayscale()

        self.update_window()

        if self.do_resize:
            self.st_flash = cv2.resize(self.st_flash, (self.resize_input_dims[0], self.resize_input_dims[1]))

        # self.st_flash = np.stack((self.st_flash, self.st_flash, self.st_flash, self.st_flash),
        #                              axis=2)

        st_flash_start_list = [self.st_flash for i in range(self.input_ch_num)]
        self.st_flash = np.stack(st_flash_start_list, axis=2)

        self.st_flash = self.st_flash.reshape(1, self.st_flash.shape[0], self.st_flash.shape[1], self.st_flash.shape[2])

    def take_screenshoot(self):

        # self.perspective_wall_prevent_overflow()

        if snake.snake_head[0] != self.state_flash_x or snake.snake_head[1] != self.state_flash_y:
            rect = pygame.Rect(-(snake.snake_head[0] - (self.pers_of_kernel[0] * self.block_size) / 2 + self.pers_fix[0]),
                               -(snake.snake_head[1] - (self.pers_of_kernel[1] * self.block_size) / 2 + self.pers_fix[1]),
                               self.pers_of_kernel[0] * self.block_size, self.pers_of_kernel[1] * self.block_size)
            self.new_st_flash_not_rot_surface = pygame.Surface((self.pers_of_kernel[0] * self.block_size,
                                                                self.pers_of_kernel[1] * self.block_size))
            pygame.Surface.blit(self.new_st_flash_not_rot_surface, self.env_window, rect)
            self.new_st_flash = pygame.surfarray.array3d(self.new_st_flash_not_rot_surface)
            # self.new_st_flash = self.new_st_flash.swapaxes(0, 1)
            # cv2.imwrite("resimler/zresim" + str(self.pic_counter + 1) + "_not_rotated_pers.png", self.new_st_flash)

            # environment_resim = pygame.surfarray.array3d(self.env_window)
            # environment_resim = environment_resim.swapaxes(0, 1)
            # cv2.imwrite("resimler/zresim" + str(self.pic_counter + 1) + "_not_rotated_env.png", environment_resim)

            if self.gui:
                self.gui_blit()

            # # Perspektif resminin yılan hep yukarı doğru görünecek şekilde düzenlenmesi
            # self.perspective_rotate_against_snake()

            self.new_st_flash = cv2.cvtColor(self.new_st_flash, cv2.COLOR_BGR2GRAY).astype(float)
            # self.new_st_flash[self.new_st_flash == 159] = 116  # yem
            # self.new_st_flash[self.new_st_flash == 29] = 127  # duvar
            # self.new_st_flash[self.new_st_flash == 150] = 63  # yılan kafası
            # self.new_st_flash[self.new_st_flash == 73] = 127  # yılan kuyruğu

            # self.perspective_not_rotated_grayscale()

            self.update_window()

            if self.do_resize:
                self.new_st_flash = cv2.resize(self.new_st_flash, (self.resize_input_dims[0], self.resize_input_dims[1]))

            self.new_st_flash = self.new_st_flash.reshape(1, self.new_st_flash.shape[0], self.new_st_flash.shape[1], 1)
            self.new_st_flash = np.append(self.new_st_flash, self.st_flash[:, :, :, :-1], axis=3)

            self.pic_counter += 1
        self.state_flash_x = snake.snake_head[0]
        self.state_flash_y = snake.snake_head[1]


class Snake(object):

    def __init__(self):
        self.change_x = None
        self.change_y = None
        self.snake_head = [0, 0]
        self.tails = [[0, 0]]
        self.snake_head_location_random()
        self.snake_tail_location_random()
        self.tails_last = [self.tails[0][0], self.tails[0][1]]
        # self.rand = 0

    def move(self):
        self.tails_last = [self.tails[len(self.tails) - 1][0], self.tails[len(self.tails) - 1][1]]
        for i in range(len(self.tails) - 1, 0, -1):
            self.tails[i] = [self.tails[i - 1][0], self.tails[i - 1][1]]
        self.tails[0] = [self.snake_head[0], self.snake_head[1]]
        self.snake_head[0] += snake.change_x
        self.snake_head[1] += snake.change_y
        # run.reverse_move_prevent = True

    def snake_head_location_random(self):
        self.snake_head[0] = np.random.choice(np.arange(start=2 * playground_init.block_size,
                                                       stop=playground_init.env_pixel[0] - 3 * playground_init.block_size,
                                                       step=playground_init.block_size))

        self.snake_head[1] = np.random.choice(np.arange(start=2 * playground_init.block_size,
                                                       stop=playground_init.env_pixel[0] - 3 * playground_init.block_size,
                                                       step=playground_init.block_size))

    def snake_tail_location_random(self):
        rand_start_direction = np.random.choice(("sol", "sağ", "yukarı", "aşağı"))
        if rand_start_direction == "sol":
            # snake_tail_rect = pygame.Rect(self.snake_head[0] + playground_init.block_size, self.snake_head[1],
            #                               playground_init.block_size, playground_init.block_size)
            self.tails[0][0] = self.snake_head[0] + playground_init.block_size
            self.tails[0][1] = self.snake_head[1]
            self.change_x = -playground_init.block_size
            self.change_y = 0
            # print("snaketaillocation_")
            # print("snaketaillocation_" + (rand_start_direction))
        elif rand_start_direction == "sağ":
            # snake_tail_rect = pygame.Rect(self.snake_head[0] - playground_init.block_size, self.snake_head[1],
            #                               playground_init.block_size, playground_init.block_size)
            self.tails[0][0] = self.snake_head[0] - playground_init.block_size
            self.tails[0][1] = self.snake_head[1]
            self.change_x = playground_init.block_size
            self.change_y = 0
            # print("snaketaillocation_")
            # print("snaketaillocation_" + (rand_start_direction))
        elif rand_start_direction == "yukarı":
            # snake_tail_rect = pygame.Rect(self.snake_head[0], self.snake_head[1] + playground_init.block_size,
            #                               playground_init.block_size, playground_init.block_size)
            self.tails[0][0] = self.snake_head[0]
            self.tails[0][1] = self.snake_head[1] + playground_init.block_size
            self.change_x = 0
            self.change_y = -playground_init.block_size
            # print("snaketaillocation_")
            # print("snaketaillocation_" + (rand_start_direction))
        elif rand_start_direction == "aşağı":
            # snake_tail_rect = pygame.Rect(self.snake_head[0], self.snake_head[1] - playground_init.block_size,
            #                               playground_init.block_size, playground_init.block_size)
            self.tails[0][0] = self.snake_head[0]
            self.tails[0][1] = self.snake_head[1] - playground_init.block_size
            self.change_x = 0
            self.change_y = playground_init.block_size
            # print("snaketaillocation_")
            # print("snaketaillocation_" + (rand_start_direction))
        # print("randomtail")

    def snake_head_spawn(self):
        snake_head_rect = pygame.Rect(self.snake_head[0], self.snake_head[1], playground_init.block_size,
                                      playground_init.block_size)
        pygame.draw.rect(playground_init.env_window, (255, 255, 255), snake_head_rect)
        # playground_init.update_window()

    # Adding tail if the snake eats food
    def add_tail(self):
        self.tails.append([self.tails_last[0], self.tails_last[1]])

    # Arranging snake's tail location
    def snake_tail_spawn(self):
        for i in range(0, len(self.tails)):
            snake_tail_rect = pygame.Rect(self.tails[i][0], self.tails[i][1], playground_init.block_size,
                                          playground_init.block_size)
            pygame.draw.rect(playground_init.env_window, (255, 45, 45), snake_tail_rect)
            # print("tail")

        # playground_init.update_window()


class Food(object):

    def __init__(self):
        self.food_state = True
        self.food = np.zeros((playground_init.number_of_food, 2))
        # self.food_location()
        self.reward = 0
        self.food_counter = 0
        self.foodIndex = None
        self.old_food = None

    def eat(self):
        for self.foodIndex in range(0, playground_init.number_of_food):
            if (snake.snake_head[0] == self.food[self.foodIndex][0] and snake.snake_head[1] ==
                    self.food[self.foodIndex][1]):
                run.score += 1
                self.reward = 10
                snake.add_tail()
                self.food_state = True
                self.food_location()
                self.food_spawn()
                # print("food " +str(self.reward))
                run.maxIdleStep += run.foodStepIncrease
            # print("max idle: " + str(run.maxIdleStep))

    # Arranging food spawn location
    def food_location(self):
        if self.food_state:
            if self.food_counter != 0:
                self.food[self.foodIndex][0] = np.random.choice(
                    np.arange(start=playground_init.block_size,
                              stop=playground_init.env_pixel[0] - playground_init.block_size,
                              step=playground_init.block_size))
                self.food[self.foodIndex][1] = np.random.choice(
                    np.arange(start=playground_init.block_size,
                              stop=playground_init.env_pixel[1] - playground_init.block_size,
                              step=playground_init.block_size))

            else:
                for i in range(0, playground_init.number_of_food):
                    self.food[i][0] = np.random.choice(
                        np.arange(start=playground_init.block_size,
                                  stop=playground_init.env_pixel[0] - playground_init.block_size,
                                  step=playground_init.block_size))
                    self.food[i][1] = np.random.choice(
                        np.arange(start=playground_init.block_size,
                                  stop=playground_init.env_pixel[1] - playground_init.block_size,
                                  step=playground_init.block_size))

        if self.food_state:
            if self.food_counter != 0:
                while self.checkFoodLocation(self.food[self.foodIndex][0], self.food[self.foodIndex][1]):
                    self.food[self.foodIndex][0] = np.random.choice(
                        np.arange(start=playground_init.block_size,
                                  stop=playground_init.env_pixel[0] - playground_init.block_size,
                                  step=playground_init.block_size))
                    self.food[self.foodIndex][1] = np.random.choice(
                        np.arange(start=playground_init.block_size,
                                  stop=playground_init.env_pixel[1] - playground_init.block_size,
                                  step=playground_init.block_size))
            if self.food_counter == 0:
                while self.checkFoodLocation(self.food, self.food):
                    for i in range(0, playground_init.number_of_food):
                        self.food[i][0] = np.random.choice(
                            np.arange(start=playground_init.block_size,
                                      stop=playground_init.env_pixel[0] - playground_init.block_size,
                                      step=playground_init.block_size))
                        self.food[i][1] = np.random.choice(
                            np.arange(start=playground_init.block_size,
                                      stop=playground_init.env_pixel[1] - playground_init.block_size,
                                      step=playground_init.block_size))

        self.food_state = False
        self.old_food = np.copy(self.food)

    def checkFoodLocation(self, posX, posY):
        if self.food_state:
            if self.food_counter != 0:
                if posX == snake.snake_head[0] and posY == snake.snake_head[1]:
                    return True
                for i in range(len(snake.tails)):
                    if posX == snake.tails[i][0] and posY == snake.tails[i][1]:
                        return True
                for i in range(0, playground_init.number_of_food):
                    if (self.old_food[i][0] == posX) and (self.old_food[i][1] == posY):
                        return True
            elif self.food_counter == 0:
                for i in range(0, playground_init.number_of_food):
                    if posX[i][0] == snake.snake_head[0] and posX[i][1] == snake.snake_head[1]:
                        return True

                for j in range(0, playground_init.number_of_food):
                    if posX[j][0] == snake.tails[0][0] and posX[j][1] == snake.tails[0][1]:
                        return True
                self.food_counter += 1
        return False

    def food_spawn(self):
        for i in range(0, playground_init.number_of_food):
            food_rect = pygame.Rect(self.food[i][0], self.food[i][1], playground_init.block_size,
                                    playground_init.block_size)
            pygame.draw.rect(playground_init.env_window, (70, 170, 170), food_rect)


class Run:

    def __init__(self):
        # self.run_state = True
        self.food_decay_step_counter = 1
        self.action = None
        self.done = 0
        self.episode_number = 1
        self.score = 0
        self.scores = np.array([])
        self.scores_pie_graph = np.zeros(10)
        self.scores_pie_graph_100 = np.zeros(100)
        self.highestscore = 0
        self.highestscore_counter = playground_init.number_of_food - 1
        self.highestscores = np.array([])
        self.average_score_100 = 0.0
        self.average_scores_100 = np.array([])
        self.average_scores_100_counter = 0
        self.graph_counter = 1
        self.plot_counter = 0
        self.maxIdleStep = 400  # The max step the snake can go without eating food
        self.foodStepIncrease = 100  # The number of step food gives after eaten
        self.doRestart = False
        self.highest_average_score_100 = 80

    def take_action(self, action):
        self.action = action
        # print("takeactiongirdi_action: " + str(self.action))
        # print(self.action)

        if ddqn_agent.n_actions == 3:
            # Sadece sola, sağa dönebiliyor ve ileri gidebiliyor (n_actions=3 yap)
            if snake.snake_head[0] < snake.tails[0][0]:  # sol
                if action == 0:  # aşağı
                    snake.change_x = 0
                    snake.change_y = playground_init.block_size
                elif action == 1:  # yukarı
                    snake.change_x = 0
                    snake.change_y = -playground_init.block_size
                elif action == 2:  # sol
                    snake.change_x = -playground_init.block_size
                    snake.change_y = 0
            elif snake.snake_head[0] > snake.tails[0][0]:  # sağ
                if action == 0:  # yukarı
                    snake.change_x = 0
                    snake.change_y = -playground_init.block_size
                elif action == 1:  # aşağı
                    snake.change_x = 0
                    snake.change_y = playground_init.block_size
                elif action == 2:  # sağ
                    snake.change_x = playground_init.block_size
                    snake.change_y = 0
            elif snake.snake_head[1] < snake.tails[0][1]:  # yukarı
                if action == 0:  # sol
                    snake.change_x = -playground_init.block_size
                    snake.change_y = 0
                elif action == 1:  # sağ
                    snake.change_x = playground_init.block_size
                    snake.change_y = 0
                elif action == 2:  # yukarı
                    snake.change_x = 0
                    snake.change_y = -playground_init.block_size
            elif snake.snake_head[1] > snake.tails[0][1]:  # aşağı
                if action == 0:  # sağ
                    snake.change_x = playground_init.block_size
                    snake.change_y = 0
                elif action == 1:  # sol
                    snake.change_x = -playground_init.block_size
                    snake.change_y = 0
                elif action == 2:  # aşağı
                    snake.change_x = 0
                    snake.change_y = playground_init.block_size
        elif ddqn_agent.n_actions == 4:
            # Dört yöne de gidebiliyor
            if (action == 0) and (snake.change_x != playground_init.block_size):
                snake.change_x = -playground_init.block_size
                snake.change_y = 0
            elif (action == 1) and (snake.change_x != -playground_init.block_size):
                snake.change_x = playground_init.block_size
                snake.change_y = 0
            elif (action == 2) and (snake.change_y != playground_init.block_size):
                snake.change_y = -playground_init.block_size
                snake.change_x = 0
            elif (action == 3) and (snake.change_y != -playground_init.block_size):
                snake.change_y = playground_init.block_size
                snake.change_x = 0

    # Check if the snake has hit the wall
    def hit_wall(self):
        # Restarting game if the snake has hit the wall
        if snake.snake_head[0] == 0 or snake.snake_head[1] == 0 \
                or snake.snake_head[0] == playground_init.block_size * (playground_init.board_size[0] - 1) \
                or snake.snake_head[1] == playground_init.block_size * (playground_init.board_size[1] - 1):
            food.reward = -1
            # print("hitwall"+str(food.reward))
            self.doRestart = True
            self.done = 1

    # Check if the snake has eaten itself
    def hit_itself(self):
        # if len(snake.tails) > 1 or len(snake.tails) > 1:
        for i in range(0, len(snake.tails) - 1):
            if snake.snake_head[0] == snake.tails[i][0] and snake.snake_head[1] == snake.tails[i][1]:
                # i=0
                food.reward = -1
                # print("hitself : " +str(food.reward))
                self.doRestart = True
                self.done = 1
                break

    def button_press(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if ddqn_agent.calistir == 1:
                    ddqn_agent.save_model()
                    print("QUIT. Ağırlık kaydedildi.")
                elif ddqn_agent.calistir == 2:
                    ddqn_agent.save_model()
                    print("QUIT. Ağırlık kaydedildi.")
                elif ddqn_agent.calistir == 3:
                    print("QUIT. Ağırlık kaydolmadı.")
                self.exitgame()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if ddqn_agent.calistir == 1:
                        ddqn_agent.save_model()
                        print("ESC. Ağırlık kaydedildi.")
                    elif ddqn_agent.calistir == 2:
                        ddqn_agent.save_model()
                        print("ESC. Ağırlık kaydedildi.")
                    elif ddqn_agent.calistir == 3:
                        print("ESC. Ağırlık kaydolmadı.")
                    self.exitgame()
                elif event.key == pygame.K_KP_ENTER or event.key == pygame.K_RETURN:
                    if ddqn_agent.calistir == 1:
                        print("ENTER tuşuna basıldı. Model zaten " +
                              str(ddqn_agent.replace_target) + " episodeda bir kaydedildi.")
                    elif ddqn_agent.calistir == 2:
                        os.rename(ddqn_agent.model + '.backup', ddqn_agent.model)
                        print("ENTER tuşuna basıldı. Ağırlık kaydolmadı.")
                    elif ddqn_agent.calistir == 3:
                        print("ENTER tuşuna basıldı. Ağırlık kaydolmadı.")
                    self.exitgame()
                elif event.key == pygame.K_SPACE:
                    pause_press = True
                    self.pause_game(pause_press)

    def pause_game(self, pause_press):
        while pause_press:
            for eventspace in pygame.event.get():
                if eventspace.type == pygame.KEYDOWN:
                    if eventspace.key == pygame.K_SPACE:
                        pause_press = False

    def restart(self):
        # self.done = 1
        self.doRestart = False
        self.maxIdleStep = 400
        snake.tails = [[0, 0]]
        snake.snake_head_location_random()
        snake.snake_tail_location_random()
        snake.tails_last = [snake.tails[0][0], snake.tails[0][1]]
        # self.scores_pie_graph = np.append(self.scores_pie_graph, self.score)
        index = self.episode_number % 10
        self.scores_pie_graph[index] = self.score
        index_100 = (self.episode_number-1) % 100
        self.scores_pie_graph_100[index_100] = self.score
        self.score = 0
        playground_init.env_window.fill((0, 0, 0))
        # playground_init.grid_lines()
        playground_init.wall()
        snake.snake_tail_spawn()
        snake.snake_head_spawn()
        food.food_state = True
        food.food_counter = 0
        food.food_location()
        food.food_spawn()
        playground_init.take_screenshoot()
        if ((ddqn_agent.t + 1) > playground_init.food_decay_step * self.food_decay_step_counter) and (
                playground_init.number_of_food != 1):
            playground_init.number_of_food -= 1
            self.food_decay_step_counter += 1

        average_score = self.scores_pie_graph.mean()
        self.average_score_100 = self.scores_pie_graph_100.mean()

        print("Episode Number: " + str(self.episode_number) + " Step Number: " + str(ddqn_agent.t) +
              " Highest Score: " + str(self.highestscore) +
              " Average Score: {:3.4f} ".format(average_score) +
              " Yüzlük Average Score: {:3.4f} ".format(self.average_score_100) +
              " Epsilon: {:2.4f} ".format(ddqn_agent.epsilon) +
              " Food Number: {} ".format(playground_init.number_of_food))  # + " Loss: " + str(self.avg_loss))
        ddqn_agent.episode_counter = self.episode_number
        self.episode_number += 1
        # if ddqn_agent.calistir == 1 or ddqn_agent.calistir == 2:
        #     if self.average_score_100 > self.highest_average_score_100 and playground_init.number_of_food <= 10:
        #         model_name = "67.ddqn_model_32x32_perspective16_UcResim_Food20dan1eDustu" + \
        #                      "_lr" + str(ddqn_agent.alpha) + \
        #                      "_gamma" + str(ddqn_agent.gamma) + \
        #                      "_FoodNum" + str(playground_init.number_of_food) + \
        #                      "_HundrAvSc" + str(self.average_score_100) + ".h5"
        #         ddqn_agent.q_eval.save(model_name)

    def exitgame(self):
        pygame.quit()
        sys.exit()

    def running(self):
        playground_init.env_window.fill((0, 0, 0))
        # playground_init.grid_lines()
        playground_init.wall()
        snake.snake_tail_spawn()
        snake.snake_head_spawn()
        food.food_location()
        food.food_spawn()
        playground_init.first_screenshoot()
        while True:
            self.button_press()
            playground_init.env_window.fill((0, 0, 0))
            # playground_init.grid_lines()
            playground_init.wall()

            food.food_spawn()
            self.take_action(ddqn_agent.choose_action(playground_init.st_flash))
            snake.move()
            food.eat()
            playground_init.set_highestscore()
            snake.snake_tail_spawn()
            snake.snake_head_spawn()

            self.hit_wall()
            self.hit_itself()
            playground_init.take_screenshoot()
            ddqn_agent.remember(playground_init.st_flash, self.action, food.reward, playground_init.new_st_flash,
                                self.done)
            food.reward = 0
            self.done = 0
            playground_init.st_flash = playground_init.new_st_flash
            if ddqn_agent.calistir == 1:
                ddqn_agent.learn()
                pygame.time.wait(200)
            elif ddqn_agent.calistir == 2:
                ddqn_agent.learn()
                # pygame.time.wait(100)
            else:
                pygame.time.wait(25)
                pass

            if self.doRestart and playground_init.take_graph:
                self.average_scores_100 = np.append(self.average_scores_100, self.average_score_100)
                self.average_scores_100_counter += 1
                self.highestscores = np.append(self.highestscores, self.highestscore)
                # self.scores = np.append(self.scores, self.score)

                if self.episode_number % playground_init.graph_per_episode == 0:
                    drawing_graph(self.episode_number, self.average_scores_100, self.highestscores,
                                  ddqn_agent.gamma, ddqn_agent.alpha, ddqn_agent.epsilon_dec)
            if self.doRestart:
                self.restart()

            if (ddqn_agent.t % playground_init.food_decay_step == 0 or playground_init.number_of_food <= 10) and \
                    playground_init.old_number_of_food != playground_init.number_of_food:
                self.highest_average_score_100 = 80
                playground_init.old_number_of_food = playground_init.number_of_food

            # # Loopa girmemesi için
            # self.maxIdleStep -= 1
            # if self.maxIdleStep == 0:
            #     # print("----MaxIdleStep 0 oldu----")
            #     self.restart()
            #     # food.reward = -1
            #     # self.maxIdleStep = 1000


playground_init = environment_init(env_pixel=(128, 128), board_size=(32, 32),
                                   resize_input_dims=(64, 64), pers_of_kernel=(16, 16),
                                   input_ch_num=3,
                                   number_of_food=20, food_decay_step=150000,
                                   take_graph=False, graph_per_episode=1000,
                                   gui=True)
snake = Snake()
food = Food()
run = Run()

# if __name__ == '__main__':

ddqn_agent = DDQNAgent(alpha=0.000003, gamma=0.98, n_actions=3, batch_size=32, replace_target=10,
                       input_dims=(playground_init.resize_input_dims[0], playground_init.resize_input_dims[1],
                                   playground_init.input_ch_num),
                       epsilon=1.0, epsilon_dec=(2 * 10 ** (-6)), epsilon_end=0.0001,
                       mem_size=50000, observe=2000, calistir=1,
                       fname="weights/67.ddqn_model_32x32_perspective16_UcResim_Food20dan1eDustu_lr0.000003_gamma0.99.h5")

# calistir parametresi >>>  1: Train olacak, weight yüklenmeyecek
# calistir parametresi >>>  2: Train olacak, weight yüklenir
# calistir parametresi >>>  3: Değerlendirme

# main_window = MainWindow()
run.running()

