import numpy as np
import matplotlib.pyplot as plt

def calculate_fitness(team_count, solo_count, population_size, own_tree, tvt, tvs, svt, svs):
    """
    Calculate the average fitness (expected energy gain) for the team and solo strategies.
    
    Fitness is calculated based on the probability of interacting with each strategy
    in the current population.
    
    Args:
        team_count (int): Number of agents playing the Team (Cooperation) strategy.
        solo_count (int): Number of agents playing the Solo (Competition) strategy.
        population_size (int): Total number of agents (team_count + solo_count).
        own_tree (float): Base reward for all agents (E_tree).
        tvt (float): Reward for a Team agent interacting with a Team agent (E_t|t).
        tvs (float): Reward for a Team agent interacting with a Solo agent (E_t|s).
        svt (float): Reward for a Solo agent interacting with a Team agent (E_s|t).
        svs (float): Reward for a Solo agent interacting with a Solo agent (E_s|s).
        
    Returns:
        tuple: (team_fitness, solo_fitness)
    """
    if population_size == 0:
        return 0, 0
    
    # Calculate current proportion of strategies in the population
    p_team = team_count / population_size
    p_solo = solo_count / population_size
    
    # Calculate expected fitness for a Team agent:
    # Base reward + (Prob_Team * Team_vs_Team_Reward + Prob_Solo * Team_vs_Solo_Reward)
    team_fitness = own_tree + (p_team * tvt + p_solo * tvs)
    
    # Calculate expected fitness for a Solo agent:
    # Base reward + (Prob_Team * Solo_vs_Team_Reward + Prob_Solo * Solo_vs_Solo_Reward)
    solo_fitness = own_tree + (p_team * svt + p_solo * svs)
    
    return team_fitness, solo_fitness

def reproduce(team_count, solo_count, team_fitness, solo_fitness):
    """
    Reproduce the next generation using fitness-proportional selection (Replicator Dynamics).
    The proportion of the next generation's population matches the proportion of 
    total fitness contributed by each strategy in the current generation.
    
    Args:
        team_count (int): Current number of Team agents.
        solo_count (int): Current number of Solo agents.
        team_fitness (float): Calculated fitness of Team strategy.
        solo_fitness (float): Calculated fitness of Solo strategy.
        
    Returns:
        tuple: (new_team_count, new_solo_count) for the next generation.
    """
    # Total fitness of the entire population
    total_fitness = team_count * team_fitness + solo_count * solo_fitness
    
    # Handle the case where the total fitness is zero or negative (no reproduction)
    if total_fitness <= 0:
        return team_count, solo_count
    
    population_size = team_count + solo_count
    
    # Calculate the proportion of total fitness contributed by the Team strategy
    team_proportion = (team_count * team_fitness) / total_fitness
    
    # Determine the number of new Team agents (proportional to their fitness contribution)
    new_team = int(population_size * team_proportion)
    # The rest of the population are Solo agents
    new_solo = population_size - new_team
    
    return new_team, new_solo

def run_single_simulation(initial_team, initial_solo, days, own_tree, tvt, tvs, svt, svs):
    """
    Run one full evolutionary simulation for a specified number of days.
    
    Args:
        initial_team (int): Starting number of Team agents.
        initial_solo (int): Starting number of Solo agents.
        days (int): Number of generations (days) to simulate.
        ... payoff matrix parameters ...
        
    Returns:
        list: History of teamwork frequency (proportion of Team agents) over time.
    """
    # Initialize the history with the starting frequency
    initial_pop_size = initial_team + initial_solo
    teamwork_frequency = [initial_team / initial_pop_size]
    
    team_count = initial_team
    solo_count = initial_solo
    population_size = initial_pop_size # Population size remains constant
    
    for day in range(days):
        # 1. Calculate the fitness of each strategy based on current population mix
        team_fitness, solo_fitness = calculate_fitness(team_count, solo_count, population_size, 
                                                       own_tree, tvt, tvs, svt, svs)
        
        # 2. Determine the counts for the next generation based on fitness
        team_count, solo_count = reproduce(team_count, solo_count, team_fitness, solo_fitness)
        
        # 3. Record the new teamwork frequency
        teamwork_frequency.append(team_count / population_size)
    
    return teamwork_frequency

def run_multiple_worlds(initial_team, initial_solo, days, num_worlds, own_tree, tvt, tvs, svt, svs):
    """
    Run multiple simulations (worlds) in parallel to account for initial variations 
    and show a range of possible outcomes.
    
    The initial team count for each world is randomly varied slightly (95% to 105%).
    
    Args:
        initial_team (int): Base starting number of Team agents.
        initial_solo (int): Base starting number of Solo agents.
        num_worlds (int): Number of parallel simulations to run.
        ... other simulation parameters ...
        
    Returns:
        list of lists: A list containing the frequency history for each world.
    """
    print(f"\nRunning {num_worlds} parallel worlds...")
    
    all_worlds = []
    for world_num in range(num_worlds):
        # Apply slight random variation to the initial Team count (95% to 105%)
        variation = np.random.uniform(0.95, 1.05)
        team = int(initial_team * variation)
        # Adjust solo count to keep the total population size constant
        solo = initial_solo + (initial_team - team)
        
        # Run the simulation for the world with varied initial conditions
        frequency_history = run_single_simulation(team, solo, days, own_tree, tvt, tvs, svt, svs)
        all_worlds.append(frequency_history)
        
        if (world_num + 1) % 2 == 0:
            print(f"  Completed {world_num + 1}/{num_worlds} worlds")
    
    return all_worlds

def plot_multiple_worlds(all_worlds, days):
    """
    Create a stacked area chart showing the average trend of teamwork vs. solo frequency,
    with individual world trajectories overlaid. Uses a dark, 'Primer' style theme.
    
    Args:
        all_worlds (list of lists): The frequency history for all simulations.
        days (int): The duration of the simulation.
    """
    num_worlds = len(all_worlds)
    # x-axis represents Day 0 to Day N
    day_range = np.arange(days + 1)
    
    # Convert list of lists to a numpy array for easy calculation
    all_worlds_array = np.array(all_worlds)
    # Calculate the mean teamwork frequency across all worlds for each day
    mean_teamwork = np.mean(all_worlds_array, axis=0)
    mean_solo = 1 - mean_teamwork
    
    # Create figure and axes with a dark background
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_facecolor('#3d3d3d')
    fig.patch.set_facecolor('#3d3d3d')
    
    # Plot the stacked areas for the *mean* frequencies
    # Solo (Competition) is the bottom area
    ax.fill_between(day_range, 0, mean_solo, 
                     color='#5B9BD5', alpha=0.9, label='Solo (Competition)')
    # Team (Cooperation) is the top area (from mean_solo to 1)
    ax.fill_between(day_range, mean_solo, 1, 
                     color='#E05C5C', alpha=0.9, label='Team (Cooperation)')
    
    # Overlay individual world trajectories (lines) in white
    for world in all_worlds:
        ax.plot(day_range, world, color='white', linewidth=1.2, alpha=0.7)
    
    # Styling and Labels
    ax.set_xlabel('Day', fontsize=14, fontweight='bold', color='white')
    ax.set_ylabel('Teamwork frequency', fontsize=14, fontweight='bold', color='white')
    ax.set_ylim(0, 1) # Frequency is between 0 and 1
    ax.set_xlim(0, days)
    
    # Grid and axis styling
    ax.grid(True, alpha=0.2, linestyle='-', color='white')
    ax.set_axisbelow(True)
    ax.tick_params(colors='white', labelsize=11)
    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_linewidth(2)
    
    # Legend styling
    legend = ax.legend(loc='upper right', fontsize=11, framealpha=0.8)
    legend.get_frame().set_facecolor('#2d2d2d')
    for text in legend.get_texts():
        text.set_color('white')
    
    plt.tight_layout()
    plt.show()

def print_summary_table(all_worlds, days):
    """
    Print summary statistics (min, mean, max, std dev) of teamwork frequency 
    at key checkpoints (Day 0, 25%, 50%, 75%, End).
    
    Args:
        all_worlds (list of lists): The frequency history for all simulations.
        days (int): The duration of the simulation.
    """
    all_worlds_array = np.array(all_worlds)
    
    print("\n" + "=" * 75)
    print("SUMMARY TABLE")
    print("=" * 75)
    print(f"\n{'Day':<10} {'Min Team %':<15} {'Mean Team %':<15} {'Max Team %':<15} {'Std Dev':<15}")
    print("-" * 75)
    
    # Define checkpoints for the table
    checkpoints = [0, days//4, days//2, 3*days//4, days]
    
    for day in checkpoints:
        # Get the teamwork frequency across all worlds for the current day
        day_values = all_worlds_array[:, day]
        
        # Calculate statistics, converting frequency (0-1) to percentage (0-100)
        min_val = np.min(day_values) * 100
        mean_val = np.mean(day_values) * 100
        max_val = np.max(day_values) * 100
        std_val = np.std(day_values) * 100
        
        # Print the formatted row
        print(f"{day:<10} {min_val:<15.1f} {mean_val:<15.1f} {max_val:<15.1f} {std_val:<15.2f}")

# Main program execution
if __name__ == "__main__":
    print("=" * 75)
    print("MULTI-WORLD EVOLUTION SIMULATOR (Primer Style)")
    print("=" * 75)
    print("\nExperiment with different reward structures to see how cooperation evolves!")
    print("\nSuggested scenarios to try:")
    print("  1. Prisoner's Dilemma (cooperation struggles):")
    print("     Own=2, TvT=1.0, SvS=0.5, TvS(T)=0, SvT(S)=2.0")
    print("  2. Mutualism (cooperation thrives):")
    print("     Own=2, TvT=3.0, SvS=0.5, TvS(T)=0.5, SvT(S)=1.5")
    print("  3. Primer Default (balanced):")
    print("     Own=2, TvT=1.75, SvS=0.75, TvS(T)=0.5, SvT(S)=1.5")
    
    print("\n" + "=" * 75)
    
    print("\n" + "=" * 75)
    print("CONFIGURE ENERGY REWARDS STRUCTURE")
    print("=" * 75)
    print("\nSet up the payoff matrix for your simulation:")
    print("Each agent gets: Own tree (base) + Interaction reward")
    print("\nTips:")
    print("  - Higher 'Team vs Team' encourages cooperation")
    print("  - Higher 'Solo vs Team (Solo gets)' encourages exploitation")
    print("  - If Team vs Team reward > Solo vs Team, cooperation can win!")
    
    try:
        # Get energy rewards from user, using defaults if input is empty
        print("\n--- Base Reward ---")
        OWN_TREE = float(input("Own tree reward (default 2): ") or "2")
        
        print("\n--- Interaction Rewards ---")
        print("(Hint: Try Team vs Team > Solo's exploitation to favor cooperation)")
        TEAM_VS_TEAM = float(input("Team vs Team interaction (default 1.75): ") or "1.75") # T|T
        SOLO_VS_SOLO = float(input("Solo vs Solo interaction (default 0.75): ") or "0.75") # S|S
        TEAM_VS_SOLO = float(input("Team vs Solo - Team gets (default 0.5): ") or "0.5")   # T|S
        SOLO_VS_TEAM = float(input("Team vs Solo - Solo gets (default 1.5): ") or "1.5")   # S|T
        
        # Display the configured payoff matrix and resulting total rewards
        print("\n" + "=" * 75)
        print("YOUR ENERGY REWARDS STRUCTURE")
        print("=" * 75)
        print(f"  Own tree (base): {OWN_TREE}")
        print(f"  Team vs Team:    +{TEAM_VS_TEAM} -> Total: {OWN_TREE + TEAM_VS_TEAM:.2f} each")
        print(f"  Solo vs Solo:    +{SOLO_VS_SOLO} -> Total: {OWN_TREE + SOLO_VS_SOLO:.2f} each")
        # Display total reward for T|S interaction
        print(f"  Team vs Solo:    Team +{TEAM_VS_SOLO} -> {OWN_TREE + TEAM_VS_SOLO:.2f}, Solo +{SOLO_VS_TEAM} -> {OWN_TREE + SOLO_VS_TEAM:.2f}")
        
        print("\n" + "=" * 75)
        print("SIMULATION PARAMETERS")
        print("=" * 75)
        # Get simulation parameters
        initial_team = int(input("\nEnter initial TEAM players per world: "))
        initial_solo = int(input("Enter initial SOLO players per world: "))
        days = int(input("Enter number of days to simulate (default 30): ") or "30")
        num_worlds = int(input("Enter number of worlds to simulate (default 8): ") or "8")
        
        # Validation checks
        if initial_team < 0 or initial_solo < 0:
            print("Error: Population counts must be non-negative")
        elif initial_team + initial_solo == 0:
            print("Error: Total population must be greater than 0")
        elif num_worlds < 1:
            print("Error: Must simulate at least 1 world")
        else:
            # Run the simulations
            all_worlds = run_multiple_worlds(initial_team, initial_solo, days, num_worlds,
                                            OWN_TREE, TEAM_VS_TEAM, TEAM_VS_SOLO, 
                                            SOLO_VS_TEAM, SOLO_VS_SOLO)
            
            # Print and analyze results
            print_summary_table(all_worlds, days)
            
            print("\n" + "=" * 75)
            print("FINAL RESULTS")
            print("=" * 75)
            
            # Calculate final statistics across all worlds
            final_frequencies = [world[-1] for world in all_worlds]
            mean_final = np.mean(final_frequencies)
            
            print(f"\nAcross {num_worlds} worlds:")
            print(f"  Mean final teamwork frequency: {mean_final:.3f} ({mean_final*100:.1f}%)")
            print(f"  Range: {min(final_frequencies):.3f} to {max(final_frequencies):.3f}")
            
            # Count how many worlds ended with cooperation dominating (> 50%)
            cooperation_wins = sum(1 for f in final_frequencies if f > 0.5)
            print(f"\n  Cooperation dominated in {cooperation_wins}/{num_worlds} worlds")
            
            # Simple conclusion based on mean final frequency
            if mean_final > 0.75:
                print("\n✓ Cooperation typically wins!")
            elif mean_final < 0.25:
                print("\n✗ Competition typically wins")
            else:
                print("\n≈ Mixed results - both strategies can succeed")
            
            # Display the final plot
            plot_multiple_worlds(all_worlds, days)
                
    except ValueError:
        print("Error: Please enter valid numeric values for rewards and integer values for population/days/worlds.")