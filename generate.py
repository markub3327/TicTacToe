import random
import numpy as np
import hashlib
from pettingzoo.classic import tictactoe_v3
from datasets import Dataset, Features, Value, Image, Version, Sequence
from collections import deque


# Metadata
NUM_GAMES = 10_000


def main():
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
                    "observation": [],
                    "action_mask": [],
                    "action": [],
                    "reward": [],
                    "status": [],
                },
                "player_2": {
                    "name": "Player O",
                    "frames": [],
                    "observation": [],
                    "action_mask": [],
                    "action": [],
                    "reward": [],
                    "status": [],
                }
            }

            # Reset the environment for a new game
            env.reset()

            # Render the initial frame
            frame = env.render()
            frame_queue.append(frame)

            # Run the game loop
            for agent in env.agent_iter():
                observation, reward, terminated, truncated, _ = env.last()
                state = observation['observation']
                action_mask = observation["action_mask"]
                players[agent]['reward'].append(reward)

                # Encode the current state of the game board
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
                    obs += "\n" + "+" + ("-" * 3 + "+") * 3 + "\n"
                players[agent]['observation'].append(obs)

                # Encode the action mask
                grid = action_mask.reshape(3, 3).T        
                mask = "+" + ("-" * 7 + "+") * 3 + "\n"
                for i, row in enumerate(grid):
                    mask += "| "
                    for col in row:
                        if col:
                            mask += "True" + "  | "
                        else:
                            mask += "False" + " | "
                    mask += "\n" + "+" + ("-" * 7 + "+") * 3 + "\n"
                players[agent]['action_mask'].append(mask)

                # Render the current frame
                frame = env.render()
                frame_queue.append(frame)
                players[agent]['frames'].append(
                    np.stack(frame_queue, axis=0)
                )

                # Is the game finished?
                if terminated or truncated:
                    action = None
                    if truncated:
                        status = 5
                    else:
                        # Select the winner
                        if reward == 1.0:
                            selected_player = agent
                            status = 2
                        # Select the loser
                        elif reward == -1.0:
                            losing_player = agent
                            status = 3
                        # Randomly select a player if the game is a draw
                        else:
                            selected_player = random.choice(['player_1', 'player_2'])
                            losing_player = None
                            status = 4
                    players[agent]['status'].append(status)
                else:
                    action = env.action_space(agent).sample(action_mask)
                    if len(players[agent]["status"]) == 0:
                        players[agent]["status"].append(1)
                    else:
                        players[agent]["status"].append(0)
    
                players[agent]['action'].append(action)

                # Step the environment with the selected action
                env.step(action)

            # Check that all lists for the selected player have the same length
            lengths = {k: len(v) for k, v in players[selected_player].items() if isinstance(v, list)}
            assert len(set(lengths.values())) == 1, (
                f"{players[selected_player]['name']}, GameID: {shard}: List lengths are not equal: {lengths}"
            )

            # Yield examples for the selected player
            for step in range(
                len(players[selected_player]['observation'])
            ):
                example = {
                    "messages": {
                        "game": "TicTacToe",
                        "name": players[selected_player]['name'],
                        "observation": players[selected_player]["observation"][step],
                        "action_mask": players[selected_player]['action_mask'][step],
                        "action": str(players[selected_player]['action'][step]),
                        "reward": players[selected_player]['reward'][step],
                        "status": players[selected_player]['status'][step],
                        "lives": None,  # Placeholder for lives
                        "frames_similarity": None,  # Placeholder for image embeddings
                        "info": None,  # Placeholder for additional info
                    },
                    "images": players[selected_player]['frames'][step],
                }
                yield example

            # Yield examples for the losing player if applicable
            if losing_player is not None:
                example = {
                    "messages": {
                        "game": "TicTacToe",
                        "name": players[losing_player]['name'],
                        "observation": players[losing_player]['observation'][-1],
                        "action_mask": players[losing_player]['action_mask'][-1],
                        "action": str(players[losing_player]['action'][-1]),
                        "reward": players[losing_player]['reward'][-1],
                        "status": players[losing_player]['status'][-1],
                        "lives": None,  # Placeholder for lives
                        "frames_similarity": None,  # Placeholder for image embeddings
                        "info": None,  # Placeholder for additional info
                    },
                    "images": players[losing_player]['frames'][-1],
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
                    "observation": Value("string"),
                    "action_mask": Value("string"),
                    "action": Value("string"),
                    "reward": Value("float32"),
                    "status": Value("int32"),
                    "lives": Value("int32"),
                    "frames_similarity": Sequence(Sequence(Value("float32"))),
                    "info": Value("string"),
                },
                "images": Sequence(Image()),
            }
        ),
        num_proc=64,
        gen_kwargs={"shards": shards},
    )

    # Set metadata for the dataset
    dataset.info.description = "PettingZoo TicTacToe dataset"
    dataset.info.dataset_name = "TicTacToe"
    dataset.info.citation = "https://pettingzoo.farama.org/environments/classic/tictactoe/"
    dataset.info.license = "MIT License"
    dataset.info.homepage = "https://github.com/markub3327/TicTacToe"
    dataset.info.version = Version("1.0.0")

    # Remove duplicate samples based on their hash
    keep_indices = {}
    for i, (obs, reward) in enumerate(zip(
            dataset["messages"]["observation"],
            dataset["messages"]["reward"],
            strict=True,
    )):
        if reward == -1.0:
            keep_indices[f"lose_{i}"] = i
            continue
        h = hashlib.sha3_512(obs.encode("utf-8")).hexdigest()
        if h not in keep_indices:
            keep_indices[h] = i
        else:
            print(f"Duplicate observation found for {i} sample!")
    dataset = dataset.select(list(keep_indices.values()))

    # Save the dataset
    dataset.save_to_disk(
        "/mnt/project/perun2601343/tictactoe/dataset",
        num_proc=64,
    )

if __name__ == "__main__":
    main()