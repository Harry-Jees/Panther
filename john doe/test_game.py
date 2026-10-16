#!/usr/bin/env python3
"""Quick test to launch the RPS game."""
import tkinter as tk
import sys
sys.path.insert(0, 'john doe')
from rock_paper_scissors import RPSGame

root = tk.Tk()
root.withdraw()  # Hide main window
game = RPSGame(root, ai_difficulty='medium')
root.mainloop()
print("Game ran successfully")