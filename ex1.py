import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# Utility Functions

def generate_initial_grid(N, prob_one=0.5, interactive=True):
    """
    Generates an initial NxN binary grid based on user-selected probability.
    If interactive is False, uses the provided prob_one without prompting.
    """
    if interactive:
        while True:
            print("""
Choose initial probability distribution:
1. 50% 1s, 50% 0s
2. 25% 1s, 75% 0s
3. 75% 1s, 25% 0s
            """)
            option = input("Enter choice (1-3): ")
            if option == '1':
                prob_one = 0.5
                break
            elif option == '2':
                prob_one = 0.25
                break
            elif option == '3':
                prob_one = 0.75
                break
            else:
                print("Invalid option. Please choose 1, 2 or 3.")
    return np.random.choice([0, 1], size=(N, N), p=[1 - prob_one, prob_one])


def get_blocks(grid, parity=1, wraparound=False):
    """
    Extracts 2x2 blocks from the grid based on parity and wraparound mode.
    Odd parity starts from (0,0); even starts from (1,1).
    """
    blocks = []
    N = grid.shape[0]
    offset = 0 if parity % 2 == 1 else 1
    rows = range(offset, N - 1, 2)
    cols = range(offset, N - 1, 2)

    # Get all full 2x2 blocks
    for i in rows:
        for j in cols:
            block = grid[i:i+2, j:j+2]
            if block.shape == (2, 2):
                blocks.append(((i, j), block))

    # Handle wraparound edges
    if wraparound:
        for i in rows:
            for j in [N - 1]:
                block = np.array([[grid[i % N, j % N], grid[i % N, (j+1) % N]],
                                  [grid[(i+1) % N, j % N], grid[(i+1) % N, (j+1) % N]]])
                blocks.append(((i, j), block))
        for i in [N - 1]:
            for j in cols:
                block = np.array([[grid[i % N, j % N], grid[i % N, (j+1) % N]],
                                  [grid[(i+1) % N, j % N], grid[(i+1) % N, (j+1) % N]]])
                blocks.append(((i, j), block))
        # Bottom-right corner wrap
        block = np.array([[grid[N-1, N-1], grid[N-1, 0]],
                          [grid[0, N-1], grid[0, 0]]])
        blocks.append(((N-1, N-1), block))

    return blocks

def process_block(block):
    """
    Applies update rules to a 2x2 block:
    - If 2 live cells: do nothing
    - If 0, 1, or 4: invert
    - If 3: invert and rotate 180 degrees
    """
    ones_count = np.sum(block)
    new_block = block.copy()
    if ones_count in [0, 1, 4]:
        new_block = 1 - new_block
    elif ones_count == 3:
        new_block = 1 - new_block
        new_block = np.rot90(new_block, 2)
    return new_block

def calculate_stability(prev_grid, current_grid):
    """
    Calculates percentage of unchanged cells between two generations.
    """
    unchanged = np.sum(prev_grid == current_grid)
    total = prev_grid.size
    return (unchanged / total) * 100

def display_grid(grid, generation, title_suffix=""):
    """
    Displays the grid using matplotlib with:
    - Blue solid lines: blocks for odd generations
    - Red dashed lines: blocks for even generations
    - Optional title suffix (e.g., "Glider detected")
    """
    N = grid.shape[0]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid, cmap='gray', vmin=0, vmax=1)

    # Draw block lines
    for offset, color, linestyle in [(0, 'blue', '-'), (1, 'red', '--')]:
        for i in range(offset, N, 2):
            ax.axhline(i - 0.5, color=color, linestyle=linestyle, linewidth=1.2)
            ax.axvline(i - 0.5, color=color, linestyle=linestyle, linewidth=1.2)

    # Draw border lines
    ax.axhline(-0.5, color='black', linewidth=1)
    ax.axhline(N - 0.5, color='black', linewidth=1)
    ax.axvline(-0.5, color='black', linewidth=1)
    ax.axvline(N - 0.5, color='black', linewidth=1)

    # Update title
    if title_suffix:
        ax.set_title(f"Grid State at Generation {generation} {title_suffix}")
    else:
        ax.set_title(f"Grid State at Generation {generation}")

    plt.tight_layout()
    plt.show()


# Simulation Function – Question 1
def run_simulation(N=100, generations=250, wraparound=False, show_final=True):
    """
    Runs the automaton simulation over a given number of generations.
    Tracks stability and optionally shows final grid.
    """
    grid = generate_initial_grid(N)
    prev_grid = grid.copy()
    stability_list = []

    for generation in range(1, generations + 1):
        parity = generation % 2
        blocks = get_blocks(grid, parity=parity, wraparound=wraparound)
        new_grid = grid.copy()

        # Apply rules block by block
        for (i, j), block in blocks:
            updated_block = process_block(block)
            for di in range(2):
                for dj in range(2):
                    ni = (i + di) % N if wraparound else i + di
                    nj = (j + dj) % N if wraparound else j + dj
                    if ni < N and nj < N:
                        new_grid[ni, nj] = updated_block[di, dj]

        stability = calculate_stability(prev_grid, new_grid)
        stability_list.append(stability)

        prev_grid = new_grid.copy()
        grid = new_grid

        # Print stability every 10 generations
        if generation % 10 == 0:
            print(f"Generation {generation} | Stability: {stability:.2f}%")

    # Plot stability graph
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, generations + 1), stability_list)
    plt.xlabel("Generation")
    plt.ylabel("Stability (%)")
    plt.title("System Stability Over Time")
    plt.grid()
    plt.tight_layout()
    plt.show()

    if show_final:
        display_grid(grid, generations)

# Questions 2 - gliders
def insert_custom_glider(grid, i, j):
    """
    Insert a single custom glider (based on block structure) at position (i, j).
    """
    pattern = np.zeros((6, 6), dtype=int)

    # Top left block
    pattern[0:2, 0:2] = 1
    # Top right block
    pattern[0:2, 4:6] = 1
    # Center block
    pattern[2:4, 2:4] = 1
    # Bottom left block
    pattern[4:6, 0:2] = 1
    # Bottom right block
    pattern[4:6, 4:6] = 1

    grid[i:i+6, j:j+6] = pattern


def insert_many_gliders(grid, spacing=12):
    """
    Insert multiple custom gliders diagonally across the grid.
    'spacing' controls how far apart the gliders are placed.
    """
    N = grid.shape[0]
    for n in range(0, N - 6, spacing):
        insert_custom_glider(grid, n, n)
    print(f"Inserted {n//spacing + 1} gliders along the diagonal!")
    return grid

def run_glider_simulation():
    """
    Runs a simulation with multiple custom gliders moving diagonally,
    with user-selected grid size and wraparound condition.
    """
    # Ask user for wraparound choice
    while True:
        print("""
Choose boundary conditions:
1. With wraparound 
2. Without wraparound 
        """)
        wrap_choice = input("Enter your choice (1-2): ").strip()
        if wrap_choice == '1':
            wraparound = True
            break
        elif wrap_choice == '2':
            wraparound = False
            break
        else:
            print("Invalid input. Please choose 1 or 2.")

    # Ask user for grid size
    while True:
        print("""
Choose grid size:
1. 25x25
2. 50x50
3. 100x100
        """)
        size_choice = input("Enter your choice (1-3): ").strip()
        if size_choice == '1':
            N = 25
            break
        elif size_choice == '2':
            N = 50
            break
        elif size_choice == '3':
            N = 100
            break
        else:
            print("Invalid input. Please choose 1, 2 or 3.")

    # Prepare grid
    grid = np.zeros((N, N), dtype=int)
    grid = insert_many_gliders(grid, spacing=12)

    generations = 25  # Fixed number of generations (6 images with steps of 5)

    for generation in range(generations):
        parity = generation % 2
        blocks = get_blocks(grid, parity=parity, wraparound=wraparound)
        new_grid = grid.copy()

        for (i, j), block in blocks:
            updated_block = process_block(block)
            for di in range(2):
                for dj in range(2):
                    ni = (i + di) % N if wraparound else i + di
                    nj = (j + dj) % N if wraparound else j + dj
                    if 0 <= ni < N and 0 <= nj < N:
                        new_grid[ni, nj] = updated_block[di, dj]

        # Show grid every 5 generations (for 6 pictures)
        if generation % 5 == 0:
            display_grid(new_grid, generation=generation)

        grid = new_grid.copy()


# Question 3 - Oscillator Simulation
def insert_random_oscillator(grid, force_type=None, center=False):
    N = grid.shape[0]
    oscillator_type = force_type if force_type else np.random.choice(["blinker", "real_block_oscillator"])

    if center:
        center_i = N // 2
        center_j = N // 2
    else:
        center_i = np.random.randint(2, N-2)
        center_j = np.random.randint(2, N-2)

    if oscillator_type == "blinker":
        grid[center_i-1:center_i+2, center_j] = 1
        box = (center_i-1, center_j, center_i+1, center_j)
    elif oscillator_type == "real_block_oscillator":
        pattern = np.array([
            [1,0,1,0],
            [0,1,0,1],
            [1,0,1,0],
            [0,1,0,1]
        ])
        grid[center_i-2:center_i+2, center_j-2:center_j+2] = pattern
        box = (center_i-2, center_j-2, center_i+1, center_j+1)

    return grid, oscillator_type, box

def display_grid_with_box(grid, generation, box_coords=None):
    """
    Displays the grid and optionally highlights a box (oscillator pattern).
    """
    N = grid.shape[0]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid, cmap='gray', vmin=0, vmax=1)

    # Draw block lines
    for offset, color, linestyle in [(0, 'blue', '-'), (1, 'red', '--')]:
        for i in range(offset, N, 2):
            ax.axhline(i - 0.5, color=color, linestyle=linestyle, linewidth=1.2)
            ax.axvline(i - 0.5, color=color, linestyle=linestyle, linewidth=1.2)

    # Draw border lines
    ax.axhline(-0.5, color='black', linewidth=1)
    ax.axhline(N - 0.5, color='black', linewidth=1)
    ax.axvline(-0.5, color='black', linewidth=1)
    ax.axvline(N - 0.5, color='black', linewidth=1)

    # Highlight box if given
    if box_coords is not None:
        (x1, y1, x2, y2) = box_coords
        rect = plt.Rectangle((y1-0.5, x1-0.5), y2-y1+1, x2-x1+1,
                             edgecolor='yellow', facecolor='none', linewidth=3)
        ax.add_patch(rect)

    ax.set_title(f"Grid State at Generation {generation}")
    plt.tight_layout()
    plt.show()

def run_oscillator_simulation():
    while True:
        print("""
Choose oscillator generation method:
1. Insert known oscillator (e.g., blinker, real block oscillator)
2. Random creation (random grid size and random oscillator)
3. Back to main menu
        """)
        method_choice = input("Enter your choice (1-3): ").strip()

        if method_choice == '1':
            while True:
                print("""
Choose grid size:
1. 10x10
2. 25x25
3. 50x50
4. 100x100
5. Back to previous menu
                """)
                grid_choice = input("Enter your choice (1-5): ").strip()

                if grid_choice == '1':
                    N = 10
                    break
                elif grid_choice == '2':
                    N = 25
                    break
                elif grid_choice == '3':
                    N = 50
                    break
                elif grid_choice == '4':
                    N = 100
                    break
                elif grid_choice == '5':
                    return
                else:
                    print("Invalid input. Please choose 1-5.")

            while True:
                print("""
Choose known oscillator:
1. Classic Blinker
2. Real Block Oscillator
3. Back to previous menu
                """)
                oscillator_choice = input("Enter your choice (1-3): ").strip()

                if oscillator_choice == '1':
                    oscillator_type = "blinker"
                    break
                elif oscillator_choice == '2':
                    oscillator_type = "real_block_oscillator"
                    break
                elif oscillator_choice == '3':
                    return
                else:
                    print("Invalid input. Please choose 1-3.")

            grid = np.zeros((N, N), dtype=int)
            grid, oscillator_type, box = insert_random_oscillator(grid, force_type=oscillator_type, center=True)

            print("Running known oscillator...")

        elif method_choice == '2':
            print("Running random oscillator creation...")
            N = np.random.choice([10, 25, 50, 100])
            grid = np.zeros((N, N), dtype=int)
            grid, oscillator_type, box = insert_random_oscillator(grid)
            print(f"Grid size: {N}x{N}")
            print(f"Inserted oscillator type: {oscillator_type}")

        elif method_choice == '3':
            return

        else:
            print("Invalid input. Please choose 1-3.")
            continue

        generations = 10
        history = []

        for generation in range(1, generations + 1):
            parity = generation % 2
            blocks = get_blocks(grid, parity=parity, wraparound=True)
            new_grid = grid.copy()

            for (i, j), block in blocks:
                updated_block = process_block(block)
                for di in range(2):
                    for dj in range(2):
                        ni = (i + di) % N
                        nj = (j + dj) % N
                        new_grid[ni, nj] = updated_block[di, dj]

            history.append(grid.copy())
            if len(history) > 20:
                history.pop(0)

            grid = new_grid.copy()

        for g in range(min(6, len(history))):
            display_grid_with_box(history[g], generation=g, box_coords=box)


# Main Interface
def main():
    """
    Main menu interface allowing the user to choose a simulation type.
    """
    while True:
        print("""
Main Menu:
1. Full Automaton Simulation
2. Glider Simulation
3. Oscillators and Patterns
4. Exit
        """)
        choice = input("Enter your choice (1-4): ").strip()

        if choice == '1':
            while True:
                print("""
Run Question 1 with:
1. Without wraparound
2. With wraparound
3. Back to main menu
                """)
                sub_choice = input("Enter your choice (1-3): ").strip()
                if sub_choice == '1':
                    run_simulation(N=100, generations=250, wraparound=False)
                elif sub_choice == '2':
                    run_simulation(N=100, generations=250, wraparound=True)
                elif sub_choice == '3':
                    break
                else:
                    print("Invalid input. Please choose 1, 2 or 3.")

        elif choice == '2':
            run_glider_simulation() 
        elif choice == '3':
            run_oscillator_simulation()
        elif choice == '4':
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid input. Please choose 1-4.")

if __name__ == '__main__':
    main()
