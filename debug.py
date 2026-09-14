import datetime
import cv2
import numpy as np
from datasets import load_from_disk

# Load the dataset from disk
ds = load_from_disk("./dataset")

# Create a video writer object to save the frames as a video
video_out = cv2.VideoWriter(
    filename=f"videos/tictactoe_{datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.mp4",
    fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
    fps=25,
    frameSize=(1000, 1000),
)

for row in ds:
    # Print the current agent's name
    print(f"State:\n{row['messages']['state']}")
    print(f"Action Mask:\n{row['messages']['action_mask']}")

    for frame_id, img in enumerate(row['images']):
        # Convert to BGR format for OpenCV
        frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        # Add text to the frame
        cv2.putText(
            frame,
            f"Game: {row['messages']['game']}",
            (40, 70),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (234, 232, 233),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Agent: {row['messages']['name']}",
            (40, 100),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (234, 232, 233),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Action: {row['messages']['action']}",
            (40, 130),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (234, 232, 233),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Reward: {row['messages']['reward']}",
            (40, 160),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (234, 232, 233),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Started: {row['messages']['started']}",
            (40, 190),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (234, 232, 233),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Termination: {row['messages']['termination']}",
            (40, 220),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (234, 232, 233),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Truncation: {row['messages']['truncation']}",
            (40, 250),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (234, 232, 233),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Draw: {row['messages']['draw']}",
            (40, 280),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (234, 232, 233),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"FrameID: {frame_id}",
            (40, 310),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (234, 232, 233),
            2,
            cv2.LINE_AA,
        )

        # Write the frame to the video
        video_out.write(frame)

video_out.release()