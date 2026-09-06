import imageio
from pettingzoo.classic import tictactoe_v3
import datetime

env = tictactoe_v3.env(render_mode="rgb_array")
env.reset()  # (seed=42)

frames = []

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    state = observation['observation']
    mask = observation["action_mask"]

    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample(mask)


    # Prepare the information about the current step
    agent = "Player X" if agent == "player_1" else "Player O"

    grid = mask.reshape(3, 3).T
    mask = ""
    for i, row in enumerate(grid):
        mask += " | ".join(str(v) for v in row) + "\n"
        if i < 2:
            mask += "_________" + "\n"



    print(f"Agent: {agent}")

    print("State:")
    print(state)

    print("Action Mask:")
    print(mask)

    print(f"Action: {action}")
    print(f"Reward: {reward}")
    print(f"Termination: {termination}")
    print(f"Truncation: {truncation}", "\n")

    env.step(action)

    frame = env.render()
    frames.append(frame)

env.close()

imageio.mimsave(f"tictactoe_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4", frames, fps=2)
