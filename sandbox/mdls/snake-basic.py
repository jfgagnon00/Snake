import pygame
import random
import numpy as np

from enum import Enum
from collections import namedtuple

#jeu normal

pygame.init()
#font = pygame.font.Font('arial.ttf', 25)
font = pygame.font.SysFont('arial', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 255, 0) # 0, 0, 255
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 200

class SnakeGame:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN

        # 2. move
        self._move(self.direction) # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return game_over, self.score

    def _is_collision(self):
        # hits boundary
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            return True
        # hits itself
        if self.head in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)


class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()



    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1

        # 1. check if want to quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()


        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):

        #Actions : [1, 0, 0] -> straight ; [0, 1, 0] -> right turn ; [0, 0, 1] -> left turn
        #[straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] #no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4 #(pour revenir au debut du clock_wise)
            new_dir = clock_wise[next_idx] #right-turn right -> down -> left -> up
        else: #[0, 0, 1]
            next_idx = (idx - 1) % 4 #(pour revenir au debut du clock_wise)
            new_dir = clock_wise[next_idx] #left-turn right -> up -> left -> down

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)


#agent.py
import torch
import random
import numpy as np
from collections import deque # pour stocker les infos
#from game import SnakeGameAI, Direction, Point --le premier fichier nommer game avec la classe(en haut)
#from model import Linear_QNet, QTrainer --vient de notre fichier model(en bas)
#from helper import plot

MAX_MEMORY = 100_000 # peut stocker un max de 100000 items dans le deque
BATCH_SIZE = 1000 # taille de la partie a graber dans la memoire par batch
LR = 0.001 #learning rate

class Agent:

    def __init__(self):
        self.n_games = 0 #nombre de game
        self.epsilon = 0 #controle la randomness
        self.gamma = 0.9 #discount rate doit etre une valeur en dessous de 1 habituellement environ 0.9-0.8 peut faire varier
        self.memory = deque(maxlen=MAX_MEMORY) # si on met trop ditem ca fait popleft()
        self.model = Linear_QNet(11, 256, 3) # le premier cest le size du state donc 11 possibilites, le deuxieme layer hidden du milieu on peut le modifier, le dernier est 3 car 3 valeurs de sortie pour laction
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game): #calcul letat avec les 11 possibilite
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y -20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            #Danger en avant --depend de la position courante
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            #Danger a droite
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            #Danger a gauche
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            #move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #determine food location
            game.food.x < game.head.x, #food a gauche
            game.food.x > game.head.x, #food a droite
            game.food.y < game.head.y, #food en haut
            game.food.y > game.head.y, #food en bas
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done): #done= game over
        self.memory.append((state, action, reward, next_state, done)) #deux parentese pour faire un gros tuple --- popleft si MAX_MEMORY est depasser


    def train_long_memory(self): #call ca pour loptimisation
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #list de tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample) # zip mets tous les states ensembles etc plus vite comme ca dans pytorch
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, next_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)



    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self, state):
        # random_move passe de plus en plus de  exploration a exploitation on trade un pour lautre
        self.epsilon = 80 -self.n_games  #depend du nombre de game on peut le faire varier
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2) # le 2 est inclus donc 0 1 ou 2
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0) # on lui demande de faire une prediction va faire la foward fonction dans model
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


#model.py

import torch
import torch.nn as nn
import torch.optim as optim
import os

from torch.nn import functional as F

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size): #les couches du reseau de neurone 3 couches
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './models' #cree un nouveau dossier dans notre dossier
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma): # lr = learning rate
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) # pour loptimisation utilise loptimizer adam
        self.criterion = nn.MSELoss()#une loss fonction la mean square error loss = (Qnew - Q)2
        # Q = model.predict(state0)
        # Qnew = R + v * max(Q(state1)) R = reward   v = gamma value

    def train_step(self, state, action, reward, next_state, done): # on va convertir en pytorch tensor
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # pas besoin de convertir le game over en tensor

        if len(state.shape) == 1: #(si on a juste 1 dimension (si on a juste 1 numero) on veut changer la shape)
            #(1, x)
            state = torch.unsqueeze(state, 0) # append une dimension au debut x = 0
            next_state =  torch.unsqueeze(next_state, 0)
            action =  torch.unsqueeze(action, 0)
            reward =  torch.unsqueeze(reward, 0)
            done = (done, ) # convertir une seule valeur en tuple de 2

        # 1 : predicted Q values avec le current state --  Q = model.predict(state0)
        pred = self.model(state) # 3 differentes valeurs

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action).item()] = Q_new #.item cest pour avoir une valeur et non une valeur tensor

        # 2 : nouvelle Q values avec le current state -- Qnew = R + v * max(Q(state1)) R = reward  v = gamma value --->fait ca si pas done
        # r + v * max(next_predicted Q value)-> cest juste une valeur donc pour avoir 3 valeurs comme lautre Q on fait un clone et change lindex
        #pred.clone()
        #preds[argmax(action)] = Q_new # action cest lindex du 1 dans exemple [0, 1, 0]

        self.optimizer.zero_grad() # fonction pour vider le gradient dans pytorch
        loss = self.criterion(target, pred) # target=Qnew pred=Q
        loss.backward() # apply backpropagation et update notre gradient
        self.optimizer.step()



#helper.py
import matplotlib.pyplot as plt

plt.ion()

def plot(scores, mean_scores):
    # display.clear_output(wait=True)
    # display.display(plt.gcf())
    plt.clf
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))


def train():
    plot_scores = [] #pour garder les scores
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        #get old state
        state_old = agent.get_state(game)

        #get move baser sur le current state
        final_move = agent.get_action(state_old)

        #perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        #train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        #remember store in the memory
        agent.remember(state_old, final_move, reward, state_new, done)

        if done: #done = game over
            #train de long memory (replay memory ou experience replay sentraine sur ces moves pour sameliorer)
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game:', agent.n_games, 'Score:', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()







# if __name__ == '__main__':
#     game = SnakeGame()

#     # game loop
#     while True:
#         game_over, score = game.play_step()

#         if game_over == True:
#             break

#     print('Final Score', score)


#     pygame.quit()