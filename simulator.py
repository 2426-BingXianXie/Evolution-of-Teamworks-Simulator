import numpy as np
import matplotlib.pyplot as plt

def calculate_fitness(team_count, solo_count, population_size, own_tree, tvt, tvs, svt, svs):
    """Calculate average fitness for team and solo strategies"""
    if population_size == 0:
        return 0, 0
    
    p_team = team_count / population_size
    p_solo = solo_count / population_size
    
    team_fitness = own_tree + (p_team * tvt + p_solo * tvs)
    solo_fitness = own_tree + (p_team * svt + p_solo * svs)
    
    return team_fitness, solo_fitness

def reproduce(team_count, solo_count, team_fitness, solo_fitness):
    """Reproduce based on fitness proportional selection"""
    total_fitness = team_count * team_fitness + solo_count * solo_fitness
    
    if total_fitness <= 0:
        return team_count, solo_count
    
    population_size = team_count + solo_count
    team_proportion = (team_count * team_fitness) / total_fitness
    
    new_team = int(population_size * team_proportion)
    new_solo = population_size - new_team
    
    return new_team, new_solo

def run_single_simulation(initial_team, initial_solo, days, own_tree, tvt, tvs, svt, svs):
    """Run one simulation and return teamwork frequency history"""
    teamwork_frequency = [initial_team / (initial_team + initial_solo)]
    
    team_count = initial_team
    solo_count = initial_solo
    population_size = team_count + solo_count
    
    for day in range(days):
        team_fitness, solo_fitness = calculate_fitness(team_count, solo_count, population_size, 
                                                       own_tree, tvt, tvs, svt, svs)
        team_count, solo_count = reproduce(team_count, solo_count, team_fitness, solo_fitness)
        teamwork_frequency.append(team_count / population_size)
    
    return teamwork_frequency

def run_multiple_worlds(initial_team, initial_solo, days, num_worlds, own_tree, tvt, tvs, svt, svs):
    """Run multiple simulations (worlds) in parallel"""
    print(f"\nRunning {num_worlds} parallel worlds...")
    
    all_worlds = []
    for world_num in range(num_worlds):
        variation = np.random.uniform(0.95, 1.05)
        team = int(initial_team * variation)
        solo = initial_solo + (initial_team - team)
        
        frequency_history = run_single_simulation(team, solo, days, own_tree, tvt, tvs, svt, svs)
        all_worlds.append(frequency_history)
        
        if (world_num + 1) % 2 == 0:
            print(f"  Completed {world_num + 1}/{num_worlds} worlds")
    
    return all_worlds

def plot_multiple_worlds(all_worlds, days):
    """Create stacked area chart with multiple world trajectories"""
    num_worlds = len(all_worlds)
    day_range = np.arange(days + 1)
    
    # Calculate mean teamwork frequency across all worlds
    all_worlds_array = np.array(all_worlds)
    mean_teamwork = np.mean(all_worlds_array, axis=0)
    mean_solo = 1 - mean_teamwork
    
    # Create figure with dark background like Primer
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_facecolor('#3d3d3d')
    fig.patch.set_facecolor('#3d3d3d')
    
    # Plot stacked areas (Team = red/pink, Solo = blue)
    ax.fill_between(day_range, 0, mean_solo, 
                     color='#5B9BD5', alpha=0.9, label='Solo (Competition)')
    ax.fill_between(day_range, mean_solo, 1, 
                     color='#E05C5C', alpha=0.9, label='Team (Cooperation)')
    
    # Overlay individual world trajectories in white
    for world in all_worlds:
        ax.plot(day_range, world, color='white', linewidth=1.2, alpha=0.7)
    
    # Styling
    ax.set_xlabel('Day', fontsize=14, fontweight='bold', color='white')
    ax.set_ylabel('Teamwork frequency', fontsize=14, fontweight='bold', color='white')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, days)
    
    # Grid
    ax.grid(True, alpha=0.2, linestyle='-', color='white')
    ax.set_axisbelow(True)
    
    # Tick styling
    ax.tick_params(colors='white', labelsize=11)
    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_linewidth(2)
    
    # Legend
    legend = ax.legend(loc='upper right', fontsize=11, framealpha=0.8)
    legend.get_frame().set_facecolor('#2d2d2d')
    for text in legend.get_texts():
        text.set_color('white')
    
    plt.tight_layout()
    plt.show()

def print_summary_table(all_worlds, days):
    """Print summary statistics table"""
    all_worlds_array = np.array(all_worlds)
    
    print("\n" + "=" * 75)
    print("SUMMARY TABLE")
    print("=" * 75)
    print(f"\n{'Day':<10} {'Min Team %':<15} {'Mean Team %':<15} {'Max Team %':<15} {'Std Dev':<15}")
    print("-" * 75)
    
    checkpoints = [0, days//4, days//2, 3*days//4, days]
    
    for day in checkpoints:
        day_values = all_worlds_array[:, day]
        min_val = np.min(day_values) * 100
        mean_val = np.mean(day_values) * 100
        max_val = np.max(day_values) * 100
        std_val = np.std(day_values) * 100
        
        print(f"{day:<10} {min_val:<15.1f} {mean_val:<15.1f} {max_val:<15.1f} {std_val:<15.2f}")

# Main program
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
        # Get energy rewards from user
        print("\n--- Base Reward ---")
        OWN_TREE = float(input("Own tree reward (default 2): ") or "2")
        
        print("\n--- Interaction Rewards ---")
        print("(Hint: Try Team vs Team > Solo's exploitation to favor cooperation)")
        TEAM_VS_TEAM = float(input("Team vs Team interaction (default 1.75): ") or "1.75")
        SOLO_VS_SOLO = float(input("Solo vs Solo interaction (default 0.75): ") or "0.75")
        TEAM_VS_SOLO = float(input("Team vs Solo - Team gets (default 0.5): ") or "0.5")
        SOLO_VS_TEAM = float(input("Team vs Solo - Solo gets (default 1.5): ") or "1.5")
        
        print("\n" + "=" * 75)
        print("YOUR ENERGY REWARDS STRUCTURE")
        print("=" * 75)
        print(f"  Own tree (base): {OWN_TREE}")
        print(f"  Team vs Team:    +{TEAM_VS_TEAM} -> Total: {OWN_TREE + TEAM_VS_TEAM:.2f} each")
        print(f"  Solo vs Solo:    +{SOLO_VS_SOLO} -> Total: {OWN_TREE + SOLO_VS_SOLO:.2f} each")
        print(f"  Team vs Solo:    Team +{TEAM_VS_SOLO} -> {OWN_TREE + TEAM_VS_SOLO:.2f}, Solo +{SOLO_VS_TEAM} -> {OWN_TREE + SOLO_VS_TEAM:.2f}")
        
        print("\n" + "=" * 75)
        print("SIMULATION PARAMETERS")
        print("=" * 75)
        initial_team = int(input("\nEnter initial TEAM players per world: "))
        initial_solo = int(input("Enter initial SOLO players per world: "))
        days = int(input("Enter number of days to simulate (default 30): ") or "30")
        num_worlds = int(input("Enter number of worlds to simulate (default 8): ") or "8")
        
        if initial_team < 0 or initial_solo < 0:
            print("Error: Population counts must be non-negative")
        elif initial_team + initial_solo == 0:
            print("Error: Total population must be greater than 0")
        elif num_worlds < 1:
            print("Error: Must simulate at least 1 world")
        else:
            # Run multiple worlds
            all_worlds = run_multiple_worlds(initial_team, initial_solo, days, num_worlds,
                                            OWN_TREE, TEAM_VS_TEAM, TEAM_VS_SOLO, 
                                            SOLO_VS_TEAM, SOLO_VS_SOLO)
            
            # Print summary table
            print_summary_table(all_worlds, days)
            
            # Analysis
            print("\n" + "=" * 75)
            print("FINAL RESULTS")
            print("=" * 75)
            
            final_frequencies = [world[-1] for world in all_worlds]
            mean_final = np.mean(final_frequencies)
            
            print(f"\nAcross {num_worlds} worlds:")
            print(f"  Mean final teamwork frequency: {mean_final:.3f} ({mean_final*100:.1f}%)")
            print(f"  Range: {min(final_frequencies):.3f} to {max(final_frequencies):.3f}")
            
            cooperation_wins = sum(1 for f in final_frequencies if f > 0.5)
            print(f"\n  Cooperation dominated in {cooperation_wins}/{num_worlds} worlds")
            
            if mean_final > 0.75:
                print("\n✓ Cooperation typically wins!")
            elif mean_final < 0.25:
                print("\n✗ Competition typically wins")
            else:
                print("\n≈ Mixed results - both strategies can succeed")
            
            plot_multiple_worlds(all_worlds, days)
                
    except ValueError:
        print("Error: Please enter valid integer values")