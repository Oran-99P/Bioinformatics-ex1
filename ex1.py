import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# Utility Functions

def generate_initial_grid(N):
    """
    Generates an initial NxN binary grid based on user-selected probability.
    """
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

    # Generate random grid of 0s and 1s
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

def display_grid(grid, generation):
    """
    Displays the grid using matplotlib with:
    - Blue solid lines: blocks for odd generations
    - Red dashed lines: blocks for even generations
    """
    N = grid.shape[0]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid, cmap='gray', vmin=0, vmax=1)

    # Draw block lines for both parities
    for offset, color, linestyle in [(0, 'blue', '-'), (1, 'red', '--')]:
        for i in range(offset, N, 2):
            ax.axhline(i - 0.5, color=color, linestyle=linestyle, linewidth=1.2)
            ax.axvline(i - 0.5, color=color, linestyle=linestyle, linewidth=1.2)

    # Draw border lines
    ax.axhline(-0.5, color='black', linewidth=1)
    ax.axhline(N - 0.5, color='black', linewidth=1)
    ax.axvline(-0.5, color='black', linewidth=1)
    ax.axvline(N - 0.5, color='black', linewidth=1)

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
from itertools import product

def shift_grid(grid, dx, dy):
    """
    Shifts a grid by (dx, dy) with wraparound.
    """
    return np.roll(np.roll(grid, dy, axis=1), dx, axis=0)

def is_shifted_match(grid1, grid2):
    """
    Checks whether grid2 is a shifted version of grid1 (with wraparound).
    """
    N = grid1.shape[0]
    for dx in range(N):
        for dy in range(N):
            if np.array_equal(shift_grid(grid1, dx, dy), grid2):
                return True
    return False

def contains_pattern(grid, pattern):
    """
    Checks if 'pattern' exists anywhere in 'grid' (no wraparound).
    """
    n, m = grid.shape
    p, q = pattern.shape
    for i in range(n - p + 1):
        for j in range(m - q + 1):
            if np.array_equal(grid[i:i+p, j:j+q], pattern):
                return True
    return False


def search_for_glider_patterns_3x3(max_generations=30, wraparound=True):
    """
    Tries all 512 binary 3x3 patterns (excluding all-0 and all-1),
    places each in center of 10x10 grid,
    runs automaton, and checks if any of them becomes a glider.
    """
    N = 10
    MAX_TRIES = 1000
    tries = 0

    print(f"Scanning 3x3 patterns (wraparound = {'on' if wraparound else 'off'})...")

    for bits in product([0, 1], repeat=9):
        if tries >= MAX_TRIES:
            print("Too many attempts without finding glider.")
            break

        pattern = np.array(bits, dtype=int).reshape((3, 3))
        if np.sum(pattern) == 0 or np.sum(pattern) == 9:
            continue

        grid = np.zeros((N, N), dtype=int)
        grid[3:6, 3:6] = pattern
        history = []

        for generation in range(1, max_generations + 1):
            parity = generation % 2
            blocks = get_blocks(grid, parity=parity, wraparound=wraparound)
            new_grid = grid.copy()

            for (i, j), block in blocks:
                updated_block = process_block(block)
                for di in range(2):
                    for dj in range(2):
                        ni = (i + di) % N if wraparound else i + di
                        nj = (j + dj) % N if wraparound else j + dj
                        if ni < N and nj < N:
                            new_grid[ni, nj] = updated_block[di, dj]
            for past in history:
                if wraparound:
                    match = is_shifted_match(past, new_grid)
                else:
                    match = contains_pattern(new_grid, pattern)
                if match:
                    print("Glider detected in 3x3 pattern!")
                    display_grid(grid, generation=0)
                    display_grid(new_grid, generation=generation)
                    return pattern

            history.append(grid.copy())
            grid = new_grid.copy()

        tries += 1

    print("No glider found in 3x3 patterns.")
    return None

def search_for_glider_patterns_4x4(max_generations=30, wraparound=True):
    """
    Tries up to MAX_TRIES 4x4 binary patterns (excluding all-0 and all-1),
    places each in center of 10x10 grid,
    runs automaton, and checks if any of them behaves like a glider.
    """
    N = 10
    MAX_TRIES = 1000
    tries = 0

    print(f"Scanning 4x4 patterns (wraparound = {'on' if wraparound else 'off'})...")

    for bits in product([0, 1], repeat=16):
        if tries >= MAX_TRIES:
            print("Too many attempts without finding glider.")
            break

        pattern = np.array(bits, dtype=int).reshape((4, 4))
        if np.sum(pattern) == 0 or np.sum(pattern) == 16:
            continue

        grid = np.zeros((N, N), dtype=int)
        grid[3:7, 3:7] = pattern
        history = []

        for generation in range(1, max_generations + 1):
            parity = generation % 2
            blocks = get_blocks(grid, parity=parity, wraparound=wraparound)
            new_grid = grid.copy()

            for (i, j), block in blocks:
                updated_block = process_block(block)
                for di in range(2):
                    for dj in range(2):
                        ni = (i + di) % N if wraparound else i + di
                        nj = (j + dj) % N if wraparound else j + dj
                        if ni < N and nj < N:
                            new_grid[ni, nj] = updated_block[di, dj]

            for past in history:
                if wraparound:
                    match = is_shifted_match(past, new_grid)
                else:
                    match = contains_pattern(new_grid, pattern)
                if match:
                    print("Glider detected in 4x4 pattern!")
                    display_grid(grid, generation=0)
                    display_grid(new_grid, generation=generation)
                    return pattern

            history.append(grid.copy())
            grid = new_grid.copy()

        tries += 1

    print("No glider found in 4x4 patterns.")
    return None


# Question 3 - Oscillator Simulation
def run_oscillator_simulation():
    """
    Runs oscillator simulation (Q3).
    Shows patterns that repeat every few generations in-place.
    Wraparound is disabled.
    """
    N = 20
    grid = np.zeros((N, N), dtype=int)

    # Oscillator pattern (starts as vertical bar)
    # It will flip every generation into horizontal
    oscillator = np.array([
        [1, 1],
        [0, 0]
    ])
    grid[5:7, 5:7] = oscillator

    # Add a second one to make it interesting
    oscillator2 = np.array([
        [0, 0],
        [1, 1]
    ])
    grid[10:12, 10:12] = oscillator2

    generations = 20
    for generation in range(1, generations + 1):
        parity = generation % 2
        blocks = get_blocks(grid, parity=parity, wraparound=False)
        new_grid = grid.copy()

        for (i, j), block in blocks:
            updated_block = process_block(block)
            for di in range(2):
                for dj in range(2):
                    ni = i + di
                    nj = j + dj
                    if ni < N and nj < N:
                        new_grid[ni, nj] = updated_block[di, dj]

        grid = new_grid.copy()

        if generation % 2 == 0 or generation == generations:
            print(f"[Oscillator Simulation] Generation {generation}")
            display_grid(grid, generation)

############################################
# Main Interface
############################################

def main():
    """
    Main menu interface allowing user to choose simulation type.
    """
    while True:
        print("""
Main Menu:
1. Question 1 – Full Automaton Simulation
2. Question 2 – Glider Simulation
3. Question 3 – Oscillators and Patterns
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
            while True:
                print("""
                Glider Pattern Search (Q2):
                1. Search all 3x3 patterns
                2. Search all 4x4 patterns
                3. Back to main menu
                """)
                sub_choice = input("Enter your choice (1-3): ").strip()

                if sub_choice == '1':
                    while True:
                        wrap_input = input("Use wraparound? (y/n): ").strip().lower()
                        if wrap_input in ['y', 'n']:
                            wrap = (wrap_input == 'y')
                            break
                        else:
                            print("Invalid input. Please enter 'y' or 'n'.")
                    search_for_glider_patterns_3x3(wraparound=wrap)

                elif sub_choice == '2':
                    while True:
                        wrap_input = input("Use wraparound? (y/n): ").strip().lower()
                        if wrap_input in ['y', 'n']:
                            wrap = (wrap_input == 'y')
                            break
                        else:
                            print("Invalid input. Please enter 'y' or 'n'.")
                    search_for_glider_patterns_4x4(wraparound=wrap)

                elif sub_choice == '3':
                    break
                else:
                    print("Invalid input. Please choose 1, 2 or 3.")

        elif choice == '3':
            run_oscillator_simulation()

        elif choice == '4':
            print("Exiting program. Goodbye!")
            break

        else:
            print("Invalid input. Please choose 1, 2, 3 or 4.")

if __name__ == '__main__':
    main()
