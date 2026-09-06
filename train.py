import imageio
import datetime
import numpy as np
from pettingzoo.classic import tictactoe_v3
from datasets import Dataset, Features, Value, Image, Sequence

# Init agents
players = {
    "player_1": {
        "name": "Player X",
        "state": [],
        "mask": [],
        "action": [],
        "reward": [],
        "termination": [],
        "truncation": [],
    },
    "player_2": {
        "name": "Player O",
        "state": [],
        "mask": [],
        "action": [],
        "reward": [],
        "termination": [],
        "truncation": [],
    }
}

env = tictactoe_v3.env(render_mode="rgb_array")
env.reset()

frames = []

for agent in env.agent_iter():
    observation, reward, termination, truncation, _ = env.last()
    state = observation['observation']
    mask = observation["action_mask"]

    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample(mask)

    # Print the current agent's name
    print(f"Agent: {players[agent]['name']}")

    # Print the current state of the game board
    print("Observation:")
    symbols = np.full((state.shape[0], state.shape[1]), ' ', dtype=str)
    if agent == "player_1":
        symbols[state[:, :, 0] == 1] = 'X'
        symbols[state[:, :, 1] == 1] = 'O'
    elif agent == "player_2":
        symbols[state[:, :, 0] == 1] = 'O'
        symbols[state[:, :, 1] == 1] = 'X'
    obs = "+" + ("-" * 3 + "+") * 3 + "\n"
    for i, row in enumerate(symbols.T):
        obs += "| "
        for col in row:
            obs += col + " | "
        obs += "\n"
        if i < 2:
            obs += "+" + ("-" * 3 + "+") * 3 + "\n"
    obs += "+" + ("-" * 3 + "+") * 3 + "\n"
    players[agent]['state'].append(obs)
    print(obs)

    # Print the action mask
    print("Action Mask:")
    grid = mask.reshape(3, 3).T        
    mask = "+" + ("-" * 7 + "+") * 3 + "\n"
    for i, row in enumerate(grid):
        mask += "| "
        for col in row:
            if col:
                mask += "True" + "  | "
            else:
                mask += "False" + " | "
        mask += "\n"
        if i < 2:
            mask += "+" + ("-" * 7 + "+") * 3 + "\n"
    mask += "+" + ("-" * 7 + "+") * 3 + "\n"
    players[agent]['mask'].append(mask)
    print(mask)

    # Print the action taken by the agent
    players[agent]['action'].append(action)
    print(f"Action: {action}")

    players[agent]['reward'].append(reward)
    print(f"Reward: {reward}")

    players[agent]['termination'].append(termination)
    print(f"Termination: {termination}")

    players[agent]['truncation'].append(truncation)
    print(f"Truncation: {truncation}", "\n")

    env.step(action)

    frame = env.render()
    frames.append(frame)

env.close()

imageio.mimsave(f"videos/tictactoe_{datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.mp4", frames, fps=2)
