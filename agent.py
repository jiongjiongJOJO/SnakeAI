import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import LineQNet, QTrainer

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent(object):
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LineQNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        # 获取当前蛇头的位置
        head = game.snake[0]
        # 提取蛇头上下左右4个方向的x,y
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        # 当前蛇正在移动的方向
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """返回当前要走的方向"""

        final_move = [0, 0, 0]

        self.epsilon = 80 - self.n_games
        if random.randint(0, 200) < self.epsilon:  # 刚开始游戏时，尽量让其随机走
            idx = random.randint(0, 2)
            final_move[idx] = 1
        else:  # 随着游戏的进行 尽可能让其自动走
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            idx = torch.argmax(prediction).item()
            final_move[idx] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # 获取旧的状态
        state_old = agent.get_state(game)

        # 获取要移动的信息
        final_move = agent.get_action(state_old)

        # 移动，以及获取新的状态
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # 训练短暂的训练
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # 保存短暂的数据
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:  # 如果游戏结束，则进行长期训练
            # 进行长期的训练
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                game.record = record
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)


if __name__ == '__main__':
    train()
