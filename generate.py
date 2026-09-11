import random
import numpy as np
from pettingzoo.classic import tictactoe_v3
from datasets import Dataset, Features, Value, Image, Version, Sequence
from collections import deque


def main():
    # Game rounds
    NUM_GAMES = 5000

    def dataset_generator(shards):
        # Init environment
        env = tictactoe_v3.env(render_mode="rgb_array")
        for shard in shards:
            # Initialize the frame queue
            frame_queue = deque(maxlen=2)

            # Init agents
            players = {
                "player_1": {
                    "name": "Player X",
                    "frames": [],
                    "state": [],
                    "mask": [],
                    "action": [],
                    "reward": [],
                    "termination": [],
                    "truncation": [],
                    "started": [True],
                },
                "player_2": {
                    "name": "Player O",
                    "frames": [],
                    "state": [],
                    "mask": [],
                    "action": [],
                    "reward": [],
                    "termination": [],
                    "truncation": [],
                    "started": [True],
                }
            }

            # Reset the environment for a new game
            env.reset()

            # Render the initial frame
            frame = env.render()
            frame_queue.append(frame)

            # Run the game loop
            for agent in env.agent_iter():
                observation, reward, termination, truncation, _ = env.last()
                state = observation['observation']
                mask = observation["action_mask"]

                # Render the current frame
                frame = env.render()
                frame_queue.append(frame)
                players[agent]['frames'].append(
                    np.stack(frame_queue, axis=0)
                )

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

            print(f"GameID {shard}: Finish player {selected_player} with reward {players[selected_player]['reward'][-1]}")

            for step in range(
                len(players[selected_player]['state'])
            ):
                example = {
                    "messages": {
                        "game": "TicTacToe",
                        "name": players[selected_player]['name'],
                        "state": players[selected_player]['state'][step],
                        "action_mask": players[selected_player]['mask'][step],
                        "action": str(players[selected_player]['action'][step]),
                        "reward": players[selected_player]['reward'][step],
                        "termination": players[selected_player]['termination'][step],
                        "truncation": players[selected_player]['truncation'][step],
                        "started": players[selected_player]['started'][step],
                    },
                    "images": players[selected_player]['frames'][step],
                }

                yield example

        env.close()

    # Define the dataset features
    shards = list(range(NUM_GAMES))
    dataset = Dataset.from_generator(
        dataset_generator,
        features=Features(
            {
                "messages": {
                    "game": Value("string"),
                    "name": Value("string"),
                    "state": Value("string"),
                    "action_mask": Value("string"),
                    "action": Value("string"),
                    "reward": Value("float32"),
                    "termination": Value("bool"),
                    "truncation": Value("bool"),
                    "started": Value("bool"),
                },
                "images": Sequence(Image()),
            }
        ),
        num_proc=32,
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
    dataset.save_to_disk(
        "/mnt/project/perun2601343/tictactoe/dataset",
        num_proc=32,
    )

if __name__ == "__main__":
    main()