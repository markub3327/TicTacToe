# Tic-Tac-Toe

![Preview](./img/tictactoe.gif)

Tic-Tac-Toe is a simple turn-based strategy game for two players, **X** and **O**. Players take turns marking empty spaces on a **3×3 grid**, aiming to create a line of three of their symbols.

A player wins by placing **three identical marks in a row**, either **horizontally**, **vertically**, or **diagonally**. If all spaces on the board are filled and neither player has formed a winning line, the game ends in a **draw**.

## Features

- Two-player gameplay
- 3×3 game board
- Win detection for rows, columns, and diagonals
- Draw detection
- Reward-based outcome system

## Rewards

The game uses the following reward scheme:

| Outcome | Reward |
|----------|----------|
| Winner | +1 |
| Loser | -1 |
| Draw | 0 for both players |

### Illegal Moves

Taking an illegal move immediately ends the game. In this case:

- The player who made the illegal move receives a reward of **-1**.
- All other players receive a reward of **0**.

## How to Play

1. Player **X** starts the game.
2. Players alternate turns placing their mark on an empty cell.
3. The first player to align three marks in a row, column, or diagonal wins.
4. If the board fills up without a winner, the game ends in a draw.
5. Making an illegal move immediately ends the game with the corresponding penalty.