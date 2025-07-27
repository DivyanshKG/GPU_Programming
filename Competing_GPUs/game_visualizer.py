#!/usr/bin/env python3
"""
Connect 4 Game Visualizer
A Python script to visualize and replay Connect 4 games played by GPUs.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import time
import re
from typing import List, Tuple, Optional

class Connect4Visualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Connect 4 GPU Game Visualizer")
        self.root.geometry("800x600")
        
        # Game state
        self.board_width = 7
        self.board_height = 6
        self.current_move = 0
        self.moves = []
        self.game_states = []
        
        # Colors
        self.colors = {
            'empty': '#f0f0f0',
            'player1': '#ff6b6b',  # Red for GPU1
            'player2': '#4ecdc4',  # Teal for GPU2
            'highlight': '#ffd93d',
            'grid': '#333333'
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Load button
        ttk.Button(control_frame, text="Load Game File", command=self.load_game).pack(side=tk.LEFT, padx=5)
        
        # Playback controls
        ttk.Button(control_frame, text="⏮", command=self.first_move, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="⏯", command=self.play_pause, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="⏭", command=self.next_move, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="⏭⏭", command=self.last_move, width=3).pack(side=tk.LEFT, padx=2)
        
        # Speed control
        ttk.Label(control_frame, text="Speed:").pack(side=tk.LEFT, padx=(20, 5))
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(control_frame, from_=0.1, to=3.0, variable=self.speed_var, 
                               orient=tk.HORIZONTAL, length=100)
        speed_scale.pack(side=tk.LEFT, padx=5)
        
        # Move counter
        self.move_label = ttk.Label(control_frame, text="Move: 0 / 0")
        self.move_label.pack(side=tk.LEFT, padx=20)
        
        # Game board
        board_frame = ttk.LabelFrame(main_frame, text="Game Board", padding="10")
        board_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Create board canvas
        self.canvas = tk.Canvas(board_frame, width=400, height=350, bg='white')
        self.canvas.pack()
        
        # Info panel
        info_frame = ttk.LabelFrame(main_frame, text="Game Information", padding="10")
        info_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Game info text
        self.info_text = tk.Text(info_frame, width=40, height=15, wrap=tk.WORD)
        info_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready to load game")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Initialize board
        self.draw_board()
        
        # Playback state
        self.playing = False
        self.after_id = None
        
    def draw_board(self):
        """Draw the Connect 4 board"""
        self.canvas.delete("all")
        
        # Board dimensions
        cell_size = 50
        margin = 25
        board_width = self.board_width * cell_size
        board_height = self.board_height * cell_size
        
        # Draw grid
        for row in range(self.board_height + 1):
            y = margin + row * cell_size
            self.canvas.create_line(margin, y, margin + board_width, y, 
                                   fill=self.colors['grid'], width=2)
        
        for col in range(self.board_width + 1):
            x = margin + col * cell_size
            self.canvas.create_line(x, margin, x, margin + board_height, 
                                   fill=self.colors['grid'], width=2)
        
        # Draw column numbers
        for col in range(self.board_width):
            x = margin + col * cell_size + cell_size // 2
            y = margin + board_height + 20
            self.canvas.create_text(x, y, text=str(col), font=('Arial', 12, 'bold'))
        
        # Draw pieces if we have game data
        if self.game_states and self.current_move < len(self.game_states):
            state = self.game_states[self.current_move]
            for row in range(self.board_height):
                for col in range(self.board_width):
                    if state.board[row][col] != 0:
                        x = margin + col * cell_size + cell_size // 2
                        y = margin + row * cell_size + cell_size // 2
                        color = self.colors['player1'] if state.board[row][col] == 1 else self.colors['player2']
                        self.canvas.create_oval(x-20, y-20, x+20, y+20, 
                                              fill=color, outline=self.colors['grid'], width=2)
        
        # Highlight last move
        if self.moves and self.current_move > 0 and self.current_move <= len(self.moves):
            last_move = self.moves[self.current_move - 1]
            x = margin + last_move.column * cell_size + cell_size // 2
            y = margin + 5
            self.canvas.create_oval(x-25, y-25, x+25, y+25, 
                                  outline=self.colors['highlight'], width=3)
    
    def load_game(self):
        """Load a game file"""
        filename = filedialog.askopenfilename(
            title="Select Game File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.parse_game_file(filename)
                self.current_move = 0
                self.update_display()
                self.status_var.set(f"Loaded game with {len(self.moves)} moves")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load game file: {str(e)}")
    
    def parse_game_file(self, filename):
        """Parse the game replay file"""
        self.moves = []
        self.game_states = []
        
        with open(filename, 'r') as f:
            content = f.read()
        
        # Parse moves
        move_pattern = r"Move (\d+):\nPlayer: ([^\n]+)\nColumn: (\d+)\nGPU1 Time: ([^\n]+)\nGPU2 Time: ([^\n]+)\nWinner: (\d+)"
        moves = re.findall(move_pattern, content)
        
        for move_data in moves:
            move_num, player, column, gpu1_time, gpu2_time, winner = move_data
            move = Move(
                int(column),
                1 if "GPU1" in player else 2,
                float(gpu1_time.split()[0]),
                float(gpu2_time.split()[0]),
                int(winner)
            )
            self.moves.append(move)
        
        # Parse board states
        board_pattern = r"\|([ XO\|]+)\|"
        board_sections = content.split("Move")
        
        for section in board_sections[1:]:  # Skip header
            lines = section.strip().split('\n')
            board_lines = []
            for line in lines:
                if line.startswith('|') and line.endswith('|'):
                    board_lines.append(line)
            
            if len(board_lines) == self.board_height:
                state = GameState()
                for row, line in enumerate(board_lines):
                    cells = line.split('|')[1:-1]  # Remove empty strings
                    for col, cell in enumerate(cells):
                        if cell.strip() == 'X':
                            state.board[row][col] = 1
                        elif cell.strip() == 'O':
                            state.board[row][col] = 2
                        else:
                            state.board[row][col] = 0
                self.game_states.append(state)
    
    def update_display(self):
        """Update the display with current game state"""
        self.draw_board()
        self.update_info()
        self.update_move_counter()
    
    def update_info(self):
        """Update the information panel"""
        self.info_text.delete(1.0, tk.END)
        
        if not self.moves:
            self.info_text.insert(tk.END, "No game loaded.\n\nPlease load a game file to view information.")
            return
        
        if self.current_move > 0 and self.current_move <= len(self.moves):
            move = self.moves[self.current_move - 1]
            
            info = f"Move {self.current_move}\n"
            info += "=" * 20 + "\n\n"
            info += f"Player: {'GPU1 (Evaluation)' if move.player == 1 else 'GPU2 (MCTS)'}\n"
            info += f"Column: {move.column}\n"
            info += f"GPU1 Time: {move.gpu1_time:.2f} ms\n"
            info += f"GPU2 Time: {move.gpu2_time:.2f} ms\n"
            
            if move.winner != 0:
                winner_name = "GPU1 (Evaluation)" if move.winner == 1 else "GPU2 (MCTS)"
                info += f"\n🏆 WINNER: {winner_name} 🏆\n"
            
            info += "\n" + "=" * 20 + "\n\n"
            
            # Add strategy descriptions
            info += "Strategies:\n\n"
            info += "GPU1 (Evaluation):\n"
            info += "• Evaluates immediate wins and blocks\n"
            info += "• Uses position-based scoring\n"
            info += "• Prefers center columns\n\n"
            
            info += "GPU2 (MCTS):\n"
            info += "• Monte Carlo Tree Search\n"
            info += "• Random playout simulations\n"
            info += "• Balances exploration/exploitation\n"
            
            self.info_text.insert(tk.END, info)
    
    def update_move_counter(self):
        """Update the move counter display"""
        total_moves = len(self.moves) if self.moves else 0
        self.move_label.config(text=f"Move: {self.current_move} / {total_moves}")
    
    def first_move(self):
        """Go to first move"""
        self.current_move = 0
        self.update_display()
    
    def last_move(self):
        """Go to last move"""
        self.current_move = len(self.moves) if self.moves else 0
        self.update_display()
    
    def next_move(self):
        """Go to next move"""
        if self.current_move < len(self.moves):
            self.current_move += 1
            self.update_display()
    
    def previous_move(self):
        """Go to previous move"""
        if self.current_move > 0:
            self.current_move -= 1
            self.update_display()
    
    def play_pause(self):
        """Toggle play/pause"""
        if self.playing:
            self.playing = False
            if self.after_id:
                self.root.after_cancel(self.after_id)
        else:
            self.playing = True
            self.play_next()
    
    def play_next(self):
        """Play next move with delay"""
        if self.playing and self.current_move < len(self.moves):
            self.next_move()
            delay = int(1000 / self.speed_var.get())  # Convert to milliseconds
            self.after_id = self.root.after(delay, self.play_next)
        else:
            self.playing = False

class Move:
    """Simple class to represent a move"""
    def __init__(self, column, player, gpu1_time, gpu2_time, winner):
        self.column = column
        self.player = player
        self.gpu1_time = gpu1_time
        self.gpu2_time = gpu2_time
        self.winner = winner

class GameState:
    """Simple class to represent game state"""
    def __init__(self):
        self.board = [[0 for _ in range(7)] for _ in range(6)]

def main():
    root = tk.Tk()
    app = Connect4Visualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main() 