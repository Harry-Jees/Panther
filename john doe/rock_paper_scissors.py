#!/usr/bin/env python3
"""Rock-Paper-Scissors with an AI opponent."""

import tkinter as tk
from tkinter import messagebox
import random

# Game logic: what each choice beats
BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}

CHOICES = ["rock", "paper", "scissors"]


def determine_winner(player, computer):
    """Return 'player', 'computer', or 'draw'."""
    if player == computer:
        return "draw"
    if BEATS[player] == computer:
        return "player"
    return "computer"


class RPSGame:
    def __init__(self, root, ai_difficulty="medium"):
        self.root = root
        self.root.title("Rock-Paper-Scissors")
        self.root.geometry("420x480")
        self.root.resizable(False, False)
        self.root.configure(bg="#f0f0f0")

        self.ai_difficulty = ai_difficulty
        self.player_score = 0
        self.computer_score = 0

        # Score display
        self.score_label = tk.Label(
            root, text="Score: You 0 - 0 Computer", font=("Helvetica", 14, "bold"),
            bg="#f0f0f0", fg="black"
        )
        self.score_label.pack(pady=10)

        # Instructions
        self.inst_label = tk.Label(
            root, text="Choose rock, paper, or scissors:", font=("Helvetica", 12),
            bg="#f0f0f0", fg="black"
        )
        self.inst_label.pack()

        # Buttons for player choices
        self.frame = tk.Frame(root, bg="#f0f0f0")
        self.frame.pack(pady=10)

        btn_width = 12
        btn_height = 2
        tk.Button(self.frame, text="Rock", width=btn_width, height=btn_height,
                  bg="#ffcccc", fg="black",
                  command=lambda: self.play("rock")).grid(row=0, column=0, padx=6, pady=6)
        tk.Button(self.frame, text="Paper", width=btn_width, height=btn_height,
                  bg="#ffffcc", fg="black",
                  command=lambda: self.play("paper")).grid(row=0, column=1, padx=6, pady=6)
        tk.Button(self.frame, text="Scissors", width=btn_width, height=btn_height,
                  bg="#ccffff", fg="black",
                  command=lambda: self.play("scissors")).grid(row=0, column=2, padx=6, pady=6)

        # Computer choice label (initially hidden)
        self.comp_choice_label = tk.Label(
            root, text="", font=("Helvetica", 12, "bold"), bg="#f0f0f0", fg="black"
        )
        self.comp_choice_label.pack(pady=5)

        # Result message
        self.result_label = tk.Label(
            root, text="", font=("Helvetica", 14, "bold"), bg="#f0f0f0", fg="black"
        )
        self.result_label.pack(pady=5)

        # Restart button
        self.restart_btn = tk.Button(
            root, text="Play Again", font=("Helvetica", 12), bg="#ddd",
            command=self.restart_game
        )
        self.restart_btn.pack(pady=10)

        # Initially hide computer choice and result
        self.comp_choice_label.config(text="")
        self.result_label.config(text="")

    def ai_choice(self):
        """Select AI move based on difficulty."""
        if self.ai_difficulty == "easy":
            # Random move
            return random.choice(CHOICES)
        elif self.ai_difficulty == "hard":
            # Simple but slightly smarter: counter the most frequent player move
            # For a single game, just pick randomly but bias toward beating player's last choice
            # Since we don't track history, just random for simplicity
            return random.choice(CHOICES)
        else:
            # medium: random
            return random.choice(CHOICES)

    def play(self, player_choice):
        computer_choice = self.ai_choice()

        winner = determine_winner(player_choice, computer_choice)

        # Update UI
        self.comp_choice_label.config(text=f"Computer chose: {computer_choice}")

        if winner == "draw":
            msg = "It's a draw!"
            self.result_label.config(text=msg, fg="black")
        elif winner == "player":
            self.player_score += 1
            msg = "You win!"
            self.result_label.config(text=msg, fg="green")
        else:
            self.computer_score += 1
            msg = "Computer wins!"
            self.result_label.config(text=msg, fg="red")

        self.score_label.config(text=f"Score: You {self.player_score} - {self.computer_score} Computer")

        # Show restart button after a result
        self.restart_btn.config(state="normal")

    def restart_game(self):
        self.player_score = 0
        self.computer_score = 0
        self.score_label.config(text="Score: You 0 - 0 Computer")
        self.comp_choice_label.config(text="")
        self.result_label.config(text="")
        self.restart_btn.config(state="disabled")


if __name__ == "__main__":
    import sys
    root = tk.Tk()
    # Default to medium difficulty AI; change to "easy" or "hard" if desired
    game = RPSGame(root, ai_difficulty="medium")
    root.mainloop()