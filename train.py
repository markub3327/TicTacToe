import random
import numpy as np
import os
from pettingzoo.classic import tictactoe_v3
from datasets import Dataset, Features, Value, Image, Version
from tqdm import tqdm

def main():
    # Game rounds
    NUM_GAMES = 1000

    # Init environment
    env = tictactoe_v3.env(render_mode="rgb_array")

    def dataset_generator(shards):
        for shard in shards:
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
                    "frames": [],
                    "started": [True],
                },
                "player_2": {
                    "name": "Player O",
                    "state": [],
                    "mask": [],
                    "action": [],
                    "reward": [],
                    "termination": [],
                    "truncation": [],
                    "frames": [],
                    "started": [True],
                }
            }

            # Reset the environment for a new game
            env.reset()

            # Run the game loop
            for agent in tqdm(env.agent_iter(), desc="Playing Tic Tac Toe"):
                observation, reward, termination, truncation, _ = env.last()
                state = observation['observation']
                mask = observation["action_mask"]

                # Render the current frame
                frame = env.render()
                players[agent]['frames'].append(frame)

                # Is the game finished?
                if termination or truncation:
                    action = None
                    # Select the winner
                    if reward == 1:
                        selected_player = agent
                    # Randomly select a player if the game is a draw
                    elif reward == 0:
                        selected_player = random.choice(['player_1', 'player_2'])
                else:
                    action = env.action_space(agent).sample(mask)

                # Print the current state of the game board
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

                # Print the action mask
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

                players[agent]['action'].append(action)
                players[agent]['reward'].append(reward)
                players[agent]['termination'].append(termination)
                players[agent]['truncation'].append(truncation)

                # Step the environment with the selected action
                env.step(action)

                # Is the game started?
                if action is not None:
                    players[agent]['started'].append(False)

            print(f"GameID {shard}: Finish player {selected_player} with reward {reward}")

            print(f"Total frames: {len(players[selected_player]['frames'])}")
            print(f"Total states: {len(players[selected_player]['state'])}")
            print(f"Total action masks: {len(players[selected_player]['mask'])}")
            print(f"Total actions: {len(players[selected_player]['action'])}")
            print(f"Total rewards: {len(players[selected_player]['reward'])}")
            print(f"Total terminations: {len(players[selected_player]['termination'])}")
            print(f"Total truncations: {len(players[selected_player]['truncation'])}")
            print(f"Total started: {len(players[selected_player]['started'])}")

            for step in range(
                len(players[selected_player]['frames'])
            ):
                print(f"Step {step}")
        
                example = {
                    "messages": {
                        "game": "TicTacToe",
                        "name": players[selected_player]['name'],
                        "state": players[selected_player]['state'][step],
                        "action_mask": players[selected_player]['mask'][step],
                        "action": players[selected_player]['action'][step],
                        "reward": players[selected_player]['reward'][step],
                        "termination": players[selected_player]['termination'][step],
                        "truncation": players[selected_player]['truncation'][step],
                        "started": players[selected_player]['started'][step],
                    },
                    "images": players[selected_player]['frames'][step]
                }

                yield example

    env.close()

    # Define the dataset features
    shards = list(range(NUM_GAMES))
    cpus = os.cpu_count()
    dataset = Dataset.from_generator(
        dataset_generator,
        features=Features(
            {
                "messages": {
                    "game": Value("string"),
                    "name": Value("string"),
                    "state": Value("string"),
                    "action_mask": Value("string"),
                    "action": Value("int32"),
                    "reward": Value("int32"),
                    "termination": Value("bool"),
                    "truncation": Value("bool"),
                    "started": Value("bool"),
                },
                "images": Image(),
            }
        ),
        num_proc=cpus,
        gen_kwargs={"shards": shards},
    )

    # Set metadata for the dataset
    dataset.info.description = "PettingZoo TicTacToe dataset"
    dataset.info.dataset_name = "TicTacToe"
    dataset.info.citation = "https://pettingzoo.farama.org/environments/classic/tictactoe/"
    dataset.info.license = "MIT License"
    dataset.info.homepage = "https://github.com/markub3327/TicTacToe"
    dataset.info.version = Version("1.0.0")

    # Save the dataset
    ds_path = "/mnt/project/perun2601343/tictactoe/dataset"
    dataset.save_to_disk(
        os.path.join(ds_path, "tictactoe"),
        num_proc=cpus,
    )

if __name__ == "__main__":
    main()