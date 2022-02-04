import torch
import torch.nn as nn
import torch.nn.functional as F

class actor(nn.Module):
    def __init__(self,input_dims, use_goals):
        super(actor, self).__init__()
        if use_goals:
            self.fc1 = nn.Linear(input_dims['obs'] + input_dims['goal'], 256)
        else:
            self.fc1 = nn.Linear(input_dims['obs'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, input_dims['action'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = torch.tanh(self.action_out(x))

        return actions

class critic(nn.Module):
    def __init__(self,input_dims, use_goals):
        super(critic, self).__init__()
        self.max_action = input_dims['action_max']
        if use_goals:
            self.fc1 = nn.Linear(input_dims['obs'] + input_dims['goal'] + input_dims['action'], 256)
        else:
            self.fc1 = nn.Linear(input_dims['obs'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value

class ForwardDynamics(nn.Module):
    def __init__(self, input_dims, use_goals):
        super(ForwardDynamics, self).__init__()
        if use_goals:
            self.fc1 = nn.Linear(input_dims['obs'], 256)
        else:
            self.fc1 = nn.Linear(input_dims['obs'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.obs_out = nn.Linear(256, input_dims['obs'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        next_state = self.obs_out(x)

        return next_state