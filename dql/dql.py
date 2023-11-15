from collections import deque
import math
import os
import random
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


DEVICE = get_device()


class DQN(nn.Module):
    def __init__(self, n_observations: int, n_actions: int):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=n_observations, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=n_actions),
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, transition):
        """Save a transition"""
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DeepQLearning:
    def __init__(
        self,
        train_env,
        eval_env,
        name="dql",
        replay_memory_size: int = 10000,
        eps_start: float = 1,
        eps_end: float = 0,
        eps_decay: int = 1000,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
    ) -> None:
        self.name = name

        self.train_env = train_env
        self.eval_env = eval_env
        self.n_actions = train_env.action_space.n
        self.n_observations = train_env.observation_space.shape[0]

        # initialize action-value function Q with random weights
        self.action_value_net = DQN(self.n_observations, self.n_actions).to(DEVICE)
        self.target_action_value_net = DQN(self.n_observations, self.n_actions).to(
            DEVICE
        )
        self.target_action_value_net.load_state_dict(
            self.action_value_net.state_dict()
        )  # copy weights

        self.replay_memory = ReplayMemory(replay_memory_size)
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.gamma = gamma

        # initialize loss function and optimizer
        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = torch.optim.AdamW(
            self.action_value_net.parameters(), lr=learning_rate, amsgrad=True
        )

    def _compute_epsilon_decay(self, current_step):
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * current_step / self.eps_decay
        )

    def _optimize_model(self, batch_size: int = 64):
        # while memory doesn't contain at least one batch, pass
        if len(self.replay_memory) < batch_size:
            return
        batch = self.replay_memory.sample(batch_size)

        observations = torch.cat([transition[0] for transition in batch])

        # boolean mask, 0 if final state else 1
        non_final_mask = torch.tensor(
            [0 if transition[3] == None else 1 for transition in batch],
            device=DEVICE,
            dtype=torch.bool,
        )
        next_observations = torch.cat(
            [transition[3] for transition in batch if transition[3] is not None]
        )  # contains only observations that are not final states

        actions = torch.cat([transition[1] for transition in batch])
        rewards = torch.tensor([transition[2] for transition in batch], device=DEVICE)

        with torch.no_grad():
            # expected values of the next observations, according to the target network
            values_next_observations = self.target_action_value_net(next_observations)

        # expected values of the observations, according to the trained model
        values_observations = self.action_value_net(observations)

        # values of the action of the transition in the batch
        values_observation_actions = values_observations.gather(1, actions)

        # y is reward for final state, else reward + gamma * max value next observation
        y = torch.zeros(batch_size, device=DEVICE)
        y += rewards
        y[non_final_mask] += self.gamma * values_next_observations.max(1)[0]
        y = y.unsqueeze(1)

        loss = self.loss_fn(values_observation_actions, y)
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(action_value_net.parameters(), 100)    # in-place gradient clipping
        self.optimizer.step()

    def select_action(
        self,
        observation,
        current_step,
    ):
        eps = self._compute_epsilon_decay(current_step)
        # print(eps)
        sample = random.random()
        if sample > eps:  # get action from model
            return self.action_value_net(observation).max(1)[1].view(1, 1)
        else:  # sample random action
            return torch.tensor(
                [[self.train_env.action_space.sample()]],
                device=DEVICE,
                dtype=torch.long,
            )

    def train(
        self,
        n_episodes: int = 1000,
        model_update_frequency: int = 100,
        save_frequency: int = 1000,
    ):
        global_step_counter = 0
        eval_steps = []
        for _ in tqdm(range(n_episodes)):
            terminated, truncated = False, False
            observation, _ = self.train_env.reset()
            observation = torch.tensor(
                observation, device=DEVICE, dtype=torch.float32
            ).unsqueeze(
                0
            )  # turn observation into a tensor that can be operated by the network
            while not (terminated or truncated):  # perform one episode
                action = self.select_action(observation, global_step_counter)
                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = self.train_env.step(action.item())
                if terminated:
                    next_observation = None
                else:
                    next_observation = torch.tensor(
                        next_observation, device=DEVICE, dtype=torch.float32
                    ).unsqueeze(
                        0
                    )  # turn observation into a tensor that can be operated by the network

                # add transition to the replay memory
                self.replay_memory.push([observation, action, reward, next_observation])

                # train model
                self._optimize_model()

                # update the target model with the given frequency
                if global_step_counter % model_update_frequency == 0:
                    self.target_action_value_net.load_state_dict(
                        self.action_value_net.state_dict()
                    )

                # save model with the given frequency
                if global_step_counter % save_frequency == 0:
                    # name file with padded zeros
                    model_dir = os.path.join("dql/models", self.name)
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    checpkt_dir = os.path.join(model_dir, "checkpoints")
                    if not os.path.exists(checpkt_dir):
                        os.makedirs(checpkt_dir)
                    checkpoint_file = os.path.join(
                        checpkt_dir, f"{global_step_counter:08d}.pt"
                    )
                    torch.save(
                        self.target_action_value_net.state_dict(), checkpoint_file
                    )

                    # evaluate model ten times and save the average number of runs
                    mean_n_steps = 0
                    for _ in range(10):
                        n_steps = self.perform_episode(self.eval_env, truncate=True)
                        # print(n_steps)
                        mean_n_steps += n_steps
                    eval_steps += [(global_step_counter, mean_n_steps / 10)]

                global_step_counter += 1
                observation = next_observation

        # self.show_eval(eval_steps)

    def show_eval(self, eval_steps):
        plt.plot(*zip(*eval_steps))
        plt.xlabel("Training step")
        plt.ylabel("Number of steps of the evaluation episodes")
        plt.savefig(f"dql/models/{self.name}/eval.png")

    def perform_episode(self, env, truncate=False):
        terminated, truncated = False, False
        observation, _ = env.reset()
        observation = torch.tensor(
            observation, device=DEVICE, dtype=torch.float32
        ).unsqueeze(
            0
        )  # turn observation into a tensor that can be operated by the network
        n_steps = 0
        while not (terminated or (truncate and truncated)):
            # print(self.target_action_value_net(observation))
            action = (
                self.target_action_value_net(observation).max(1)[1].view(1, 1)
            )  # select the best action (maximize value) using the trained model
            next_observation, _, terminated, truncated, _ = env.step(action.item())
            next_observation = torch.tensor(
                next_observation, device=DEVICE, dtype=torch.float32
            ).unsqueeze(
                0
            )  # turn observation into a tensor that can be operated by the network
            observation = next_observation
            n_steps += 1
        env.close()
        return n_steps

    def load_weights(self, weights_path):
        self.target_action_value_net.load_state_dict(
            torch.load(weights_path, map_location=torch.device(DEVICE))
        )
