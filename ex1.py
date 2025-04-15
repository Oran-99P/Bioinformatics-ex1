import numpy as np
import matplotlib.pyplot as plt

############################################
# Utility Functions
############################################

def generate_initial_grid(N):
    """
    Generates an initial NxN grid with values 0 or 1 based on a selected probability.
    Displays a menu to choose between:
    1. 50% 1s and 50% 0s
    2. 25% 1s and 75% 0s
    3. 75% 1s and 25% 0s
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

    return np.random.choice([0, 1], size=(N, N), p=[1 - prob_one, prob_one])

def get_blocks(grid, parity=1, wraparound=False):
    blocks = []
    N = grid.shape[0]
    offset = 0 if parity % 2 == 1 else 1
    rows = range(offset, N - 1, 2)
    cols = range(offset, N - 1, 2)

    for i in rows:
        for j in cols:
            block = grid[i:i+2, j:j+2]
            if block.shape == (2, 2):
                blocks.append(((i, j), block))

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
        block = np.array([[grid[N-1, N-1], grid[N-1, 0]],
                          [grid[0, N-1], grid[0, 0]]])
        blocks.append(((N-1, N-1), block))

    return blocks

def process_block(block):
    ones_count = np.sum(block)
    new_block = block.copy()
    if ones_count in [0, 1, 4]:
        new_block = 1 - new_block
    elif ones_count == 3:
        new_block = 1 - new_block
        new_block = np.rot90(new_block, 2)
    return new_block

def calculate_stability(prev_grid, current_grid):
    unchanged = np.sum(prev_grid == current_grid)
    total = prev_grid.size
    return (unchanged / total) * 100

def display_grid(grid, generation):
    N = grid.shape[0]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid, cmap='gray', vmin=0, vmax=1)

    for offset, color, linestyle in [(0, 'blue', '-'), (1, 'red', '--')]:
        for i in range(offset, N, 2):
            ax.axhline(i - 0.5, color=color, linestyle=linestyle, linewidth=1.2)
            ax.axvline(i - 0.5, color=color, linestyle=linestyle, linewidth=1.2)

    ax.axhline(-0.5, color='black', linewidth=1)
    ax.axhline(N - 0.5, color='black', linewidth=1)
    ax.axvline(-0.5, color='black', linewidth=1)
    ax.axvline(N - 0.5, color='black', linewidth=1)

    ax.set_title(f"Grid State at Generation {generation}")
    plt.tight_layout()
    plt.show()

############################################
# Simulation Function
############################################

def run_simulation(N=100, generations=250, wraparound=False, show_final=True):
    grid = generate_initial_grid(N)
    prev_grid = grid.copy()
    stability_list = []

    for generation in range(1, generations + 1):
        parity = generation % 2
        blocks = get_blocks(grid, parity=parity, wraparound=wraparound)
        new_grid = grid.copy()

        for (i, j), block in blocks:
            updated_block = process_block(block)
            for di in range(2):
                for dj in range(2):
                    new_i = (i + di) % N if wraparound else i + di
                    new_j = (j + dj) % N if wraparound else j + dj
                    if new_i < N and new_j < N:
                        new_grid[new_i, new_j] = updated_block[di, dj]


        stability = calculate_stability(prev_grid, new_grid)
        stability_list.append(stability)

        prev_grid = new_grid.copy()
        grid = new_grid

        if generation % 10 == 0:
            print(f"Generation {generation} | Stability: {stability:.2f}%")

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

############################################
# Main Interface
############################################

def main():
    while True:
        print("""
Automaton Block Simulation Menu:
1. Run standard simulation (Question 1)
2. Run with wraparound
3. Exit
        """)
        choice = input("Enter your choice (1-3): ").strip()

        if choice == '1':
            run_simulation(N=100, generations=250, wraparound=False)
        elif choice == '2':
            run_simulation(N=100, generations=250, wraparound=True)
        elif choice == '3':
            print("Exiting simulation. Goodbye!")
            break
        else:
            print("Invalid input. Please choose 1, 2 or 3.")

if __name__ == '__main__':
    main()