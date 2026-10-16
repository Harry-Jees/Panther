#!/usr/bin/env python3
"""A simple calculator with a proper GUI using Tkinter."""

import tkinter as tk
from tkinter import messagebox


def add():
    try:
        val1 = float(entry1.get())
        val2 = float(entry2.get())
        result.set(val1 + val2)
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers")


def subtract():
    try:
        val1 = float(entry1.get())
        val2 = float(entry2.get())
        result.set(val1 - val2)
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers")


def multiply():
    try:
        val1 = float(entry1.get())
        val2 = float(entry2.get())
        result.set(val1 * val2)
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers")


def divide():
    try:
        val1 = float(entry1.get())
        val2 = float(entry2.get())
        if val2 == 0:
            messagebox.showerror("Error", "Cannot divide by zero")
        else:
            result.set(val1 / val2)
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers")


def clear():
    entry1.set("")
    entry2.set("")
    result.set("")


# Main window
root = tk.Tk()
root.title("Calculator")
root.geometry("300x250")
root.resizable(False, False)

# Variables
entry1 = tk.StringVar()
entry2 = tk.StringVar()
result = tk.StringVar()

# Frame for inputs
frame = tk.LabelFrame(root, text="Calculation", padx=10, pady=10)
frame.pack(padx=10, pady=10, fill="x")

# First number
tk.Label(frame, text="Number 1:").grid(row=0, column=0, sticky="w")
tk.Entry(frame, textvariable=entry1, width=15).grid(row=0, column=1)

# Second number
tk.Label(frame, text="Number 2:").grid(row=1, column=0, sticky="w")
tk.Entry(frame, textvariable=entry2, width=15).grid(row=1, column=1)

# Result
tk.Label(frame, text="Result:").grid(row=2, column=0, sticky="w")
tk.Entry(frame, textvariable=result, width=15, state="readonly").grid(row=2, column=1)

# Buttons frame
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# Operation buttons
tk.Button(button_frame, text="Add", width=8, command=add).grid(row=0, column=0, padx=2)
tk.Button(button_frame, text="Subtract", width=8, command=subtract).grid(row=0, column=1, padx=2)
tk.Button(button_frame, text="Multiply", width=8, command=multiply).grid(row=0, column=2, padx=2)
tk.Button(button_frame, text="Divide", width=8, command=divide).grid(row=0, column=3, padx=2)

# Clear button
tk.Button(root, text="Clear", width=8, command=clear).pack(pady=5)

# Copyright/author label
tk.Label(root, text="Calculator by John Doe", fg="gray").pack(side="bottom", pady=5)

# Start main loop
root.mainloop()