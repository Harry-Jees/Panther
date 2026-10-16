"""Tic-Tac-Toe with an improved GUI and a simple automatic opponent (AI)."""

import tkinter as tk
from tkinter import messagebox
import random

WIN_COLOR = "#ffdddd"  # light red background for winning cells
BG_COLOR = "#f0f0f0"
BTN_COLOR = "#ffffff"
X_COLOR = "#ff6666"
O_COLOR = "#66b2ff"
FONT = ("Helvetica", 24, "bold")


def check_winner(board):
    """Return list of winning indices if there is a winner, else None."""
    wins = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
        (0, 4, 8), (2, 4, 6)               # diagonals
    ]
    for a, b, c in wins:
        if board[a] == board[b] == board[c] != "":
            return [a, b, c]
    return None


def available_moves(board):
    """Return list of indices where board is empty."""
    return [i for i, cell in enumerate(board) if cell == ""]


def ai_minimax(board, maximizing):
    """Simple minimax to decide the best move for O (maximizing) vs X (minimizing)."""
    winner = check_winner(board)
    if winner:
        # evaluate outcome: O wins -> +1, X wins -> -1, draw -> 0
        return (1 if board[winner[0]] == "O" else -1 if board[winner[0]] == "X" else 0, None)
    if "" not in board:
        return (0, None)

    best_move = None
    if maximizing:
        best_score = -float('inf')
        for move in available_moves(board):
            board[move] = "O"
            score, _ = ai_minimax(board, False)
            board[move] = ""
            if score > best_score:
                best_score = score
                best_move = move
        return best_score, best_move
    else:
        best_score = float('inf')
        for move in available_moves(board):
            board[move] = "X"
            score, _ = ai_minimax(board, True)
            board[move] = ""
            if score < best_score:
                best_score = score
                best_move = move
        return best_score, best_move


def ai_random(board):
    """Choose a random available move."""
    moves = available_moves(board)
    return random.choice(moves) if moves else None


class TicTacToeGUI:
    def __init__(self, root, ai_mode=False):
        self.root = root
        self.root.title("Tic-Tac-Toe")
        self.root.geometry("460x520")
        self.root.resizable(False, False)
        self.root.configure(bg=BG_COLOR)

        self.ai_mode = ai_mode  # True if player O is automatic
        self.board = [""] * 9
        self.current_player = "X"
        self.buttons = []

        # Top label showing turn
        self.label = tk.Label(
            root, text=f"Player {self.current_player}'s turn", font=("Helvetica", 16, "bold"),
            bg=BG_COLOR, fg="black"
        )
        self.label.pack(pady=10)

        # Frame for the 3x3 grid
        self.frame = tk.Frame(root, bg=BG_COLOR)
        self.frame.pack()

        for i in range(9):
            btn = tk.Button(
                self.frame, text="", font=FONT, width=5, height=2,
                bg=BTN_COLOR, fg="black",
                activebackground="#e0e0e0",
                command=lambda i=i: self.on_button_click(i)
            )
            btn.grid(row=i // 3, column=i % 3, padx=5, pady=5)
            self.buttons.append(btn)

        # Restart button
        self.restart_btn = tk.Button(
            root, text="Restart", font=("Helvetica", 12), bg="#ddd",
            command=self.restart_game
        )
        self.restart_btn.pack(pady=10)

    def on_button_click(self, idx):
        if self.board[idx] == "":
            # Human move
            self.board[idx] = self.current_player
            color = X_COLOR if self.current_player == "X" else O_COLOR
            self.buttons[idx].config(text=self.current_player, fg=color, state="disabled", disabledforeground=color)

            # Check for winner immediately
            win_indices = check_winner(self.board)
            if win_indices:
                for i in win_indices:
                    self.buttons[i].config(bg=WIN_COLOR)
                messagebox.showinfo("Game Over", f"Player {self.current_player} wins!")
                self.restart_game()
                return
            elif "" not in self.board:
                messagebox.showinfo("Game Over", "It's a draw!")
                self.restart_game()
                return

            # Switch to opponent (AI or other player)
            self.current_player = "O" if self.current_player == "X" else "X"
            self.label.config(text=f"Player {self.current_player}'s turn")

            # If AI mode and it's O's turn, make automatic move
            if self.ai_mode and self.current_player == "O":
                self.root.after(500, self.ai_move)

    def ai_move(self):
        """Choose a move for O using minimax (or random) and apply it."""
        # Decide strategy: use minimax for decent AI, random for simplicity
        # Here we use minimax; it returns best move index.
        _, move = ai_minimax(self.board, True)  # maximizing for O
        if move is None:
            # fallback to random
            move = ai_random(self.board)

        if move is not None and self.board[move] == "":
            self.board[move] = "O"
            color = O_COLOR
            self.buttons[move].config(text="O", fg=color, state="disabled", disabledforeground=color)

            # Check winner after AI move
            win_indices = check_winner(self.board)
            if win_indices:
                for i in win_indices:
                    self.buttons[i].config(bg=WIN_COLOR)
                messagebox.showinfo("Game Over", "Player O (AI) wins!")
                self.restart_game()
                return
            elif "" not in self.board:
                messagebox.showinfo("Game Over", "It's a draw!")
                self.restart_game()
                return

            # Switch back to human player X
            self.current_player = "X"
            self.label.config(text=f"Player {self.current_player}'s turn")
        else:
            # Should not happen, but fallback
            self.current_player = "X"
            self.label.config(text=f"Player {self.current_player}'s turn")

    def restart_game(self):
        self.board = [""] * 9
        self.current_player = "X"
        self.label.config(text=f"Player {self.current_player}'s turn")
        for btn in self.buttons:
            btn.config(text="", state="normal", bg=BTN_COLOR)


if __name__ == "__main__":
    import sys
    # Enable AI opponent by default for easier testing and usage
    ai_mode = True
    root = tk.Tk()
    game = TicTacToeGUI(root, ai_mode=ai_mode)
    root.mainloop()
