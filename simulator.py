import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


class Agent:
   """An individual agent with a strategy and accumulated fitness."""
   def __init__(self, strategy):
       self.strategy = strategy  # 'team' or 'solo'
       self.fitness = 0.0
  
   def reset_fitness(self):
       """Reset fitness at the start of each generation."""
       self.fitness = 0.0
  
   def add_fitness(self, reward):
       """Add reward from an interaction."""
       self.fitness += reward


def simulate_interactions(agents, own_tree, tvt, tvs, svt, svs, interactions_per_agent):
   """
   Have each agent interact with randomly selected partners.
  
   Args:
       agents (list): List of Agent objects
       own_tree (float): Base reward all agents get
       tvt, tvs, svt, svs (float): Interaction rewards
       interactions_per_agent (int): How many interactions each agent has per generation
   """
   # Reset all fitnesses at the start of the generation
   for agent in agents:
       agent.reset_fitness()
       # Give base reward
       agent.fitness += own_tree
  
   # Each agent interacts with random partners
   for agent in agents:
       for _ in range(interactions_per_agent):
           # Pick a random partner (could be same strategy or different)
           partner = np.random.choice(agents)
          
           # Determine rewards based on strategies
           if agent.strategy == 'team' and partner.strategy == 'team':
               reward = tvt
           elif agent.strategy == 'team' and partner.strategy == 'solo':
               reward = tvs
           elif agent.strategy == 'solo' and partner.strategy == 'team':
               reward = svt
           else:  # both solo
               reward = svs
          
           # Add reward to agent's fitness
           agent.add_fitness(reward)


def reproduce_population(agents):
   """
   Create next generation using fitness-proportional selection.
  
   Args:
       agents (list): Current generation of agents
      
   Returns:
       list: New generation of agents
   """
   population_size = len(agents)
  
   # Calculate total fitness
   total_fitness = sum(agent.fitness for agent in agents)
  
   if total_fitness <= 0:
       # If no fitness, just copy current population
       return [Agent(agent.strategy) for agent in agents]
  
   # Create fitness probabilities
   probabilities = [agent.fitness / total_fitness for agent in agents]
  
   # Select parents for next generation (with replacement)
   new_generation = []
   for _ in range(population_size):
       parent = np.random.choice(agents, p=probabilities)
       new_generation.append(Agent(parent.strategy))
  
   return new_generation


def run_single_simulation(initial_team, initial_solo, days, own_tree, tvt, tvs, svt, svs,
                         interactions_per_agent=5):
   """
   Run one full agent-based simulation.
  
   Args:
       initial_team (int): Starting number of Team agents
       initial_solo (int): Starting number of Solo agents
       days (int): Number of generations to simulate
       interactions_per_agent (int): How many interactions each agent has per generation
       ... payoff parameters ...
      
   Returns:
       list: History of teamwork frequency over time
   """
   # Create initial population
   agents = ([Agent('team') for _ in range(initial_team)] +
             [Agent('solo') for _ in range(initial_solo)])
  
   # Track frequency history
   teamwork_frequency = [initial_team / len(agents)]
  
   for day in range(days):
       # 1. Simulate interactions
       simulate_interactions(agents, own_tree, tvt, tvs, svt, svs, interactions_per_agent)
      
       # 2. Reproduce based on fitness
       agents = reproduce_population(agents)
      
       # 3. Record teamwork frequency
       team_count = sum(1 for agent in agents if agent.strategy == 'team')
       teamwork_frequency.append(team_count / len(agents))
  
   return teamwork_frequency


def run_multiple_worlds(initial_team, initial_solo, days, num_worlds, own_tree, tvt, tvs, svt, svs,
                      interactions_per_agent=5):
   """
   Run multiple agent-based simulations in parallel.
   """
   print(f"\nRunning {num_worlds} parallel agent-based worlds...")
   print(f"Each agent will have {interactions_per_agent} interactions per generation")
  
   all_worlds = []
   for world_num in range(num_worlds):
       # Apply slight random variation to initial team count
       variation = np.random.uniform(0.95, 1.05)
       team = int(initial_team * variation)
       solo = initial_solo + (initial_team - team)
      
       # Run the simulation
       frequency_history = run_single_simulation(team, solo, days, own_tree, tvt, tvs, svt, svs,
                                                 interactions_per_agent)
       all_worlds.append(frequency_history)
      
       if (world_num + 1) % 2 == 0:
           print(f"  Completed {world_num + 1}/{num_worlds} worlds")
  
   return all_worlds


def plot_multiple_worlds(all_worlds, days):
   """Create visualization of all simulation worlds."""
   num_worlds = len(all_worlds)
   day_range = np.arange(days + 1)
  
   all_worlds_array = np.array(all_worlds)
   mean_teamwork = np.mean(all_worlds_array, axis=0)
   mean_solo = 1 - mean_teamwork
  
   fig, ax = plt.subplots(figsize=(14, 8))
   ax.set_facecolor('#3d3d3d')
   fig.patch.set_facecolor('#3d3d3d')
  
   # Plot stacked areas for mean frequencies
   ax.fill_between(day_range, 0, mean_solo,
                    color='#5B9BD5', alpha=0.9, label='Solo (Competition)')
   ax.fill_between(day_range, mean_solo, 1,
                    color='#E05C5C', alpha=0.9, label='Team (Cooperation)')
  
   # Overlay individual world trajectories
   for world in all_worlds:
       solo_freq = [1 - t for t in world]
       ax.plot(day_range, solo_freq, color='white', linewidth=1.2, alpha=0.7)
  
   ax.set_xlabel('Day', fontsize=14, fontweight='bold', color='white')
   ax.set_ylabel('Teamwork frequency', fontsize=14, fontweight='bold', color='white')
   ax.set_ylim(0, 1)
   ax.set_xlim(0, days)
  
   ax.grid(True, alpha=0.2, linestyle='-', color='white')
   ax.set_axisbelow(True)
   ax.tick_params(colors='white', labelsize=11)
   for spine in ax.spines.values():
       spine.set_color('white')
       spine.set_linewidth(2)
  
   legend = ax.legend(loc='upper right', fontsize=11, framealpha=0.8)
   legend.get_frame().set_facecolor('#2d2d2d')
   for text in legend.get_texts():
       text.set_color('white')
  
   plt.tight_layout()
   plt.show()


def print_summary_table(all_worlds, days):
   """Print summary statistics at key checkpoints."""
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
   print("AGENT-BASED EVOLUTION SIMULATOR")
   print("=" * 75)
   print("\nThis simulator runs actual agent interactions (not EV calculations)!")
   print("Each agent interacts with random partners and accumulates fitness.")
   print("\nSuggested scenarios to try:")
   print("  1. Prisoner's Dilemma: Own=2, TvT=1.0, SvS=0.5, TvS=0, SvT=2.0")
   print("  2. Mutualism: Own=2, TvT=3.0, SvS=0.5, TvS=0.5, SvT=1.5")
   print("  3. Balanced: Own=2, TvT=1.75, SvS=0.75, TvS=0.5, SvT=1.5")
  
   print("\n" + "=" * 75)
   print("CONFIGURE ENERGY REWARDS STRUCTURE")
   print("=" * 75)
  
   try:
       print("\n--- Base Reward ---")
       OWN_TREE = float(input("Own tree reward (default 2): ") or "2")
      
       print("\n--- Interaction Rewards ---")
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
       interactions_per_agent = int(input("Interactions per agent per generation (default 5): ") or "5")
      
       if initial_team < 0 or initial_solo < 0:
           print("Error: Population counts must be non-negative")
       elif initial_team + initial_solo == 0:
           print("Error: Total population must be greater than 0")
       elif num_worlds < 1:
           print("Error: Must simulate at least 1 world")
       else:
           all_worlds = run_multiple_worlds(initial_team, initial_solo, days, num_worlds,
                                           OWN_TREE, TEAM_VS_TEAM, TEAM_VS_SOLO,
                                           SOLO_VS_TEAM, SOLO_VS_SOLO,
                                           interactions_per_agent)
          
           print_summary_table(all_worlds, days)
          
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
       print("Error: Please enter valid numeric values.")
      

