import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import pygad
from unittest.mock import MagicMock

# --- 1. SETUP & MOCKING ---
# Mock xfoil since we are using the Neural Network surrogate
sys.modules["xfoil"] = MagicMock()
sys.modules["xfoil.model"] = MagicMock()

def ensure_package_structure():
    """Ensures necessary folders exist and adds root to path."""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(root_dir)
    folders_to_fix = [
        os.path.join(root_dir, 'data'),
        os.path.join(root_dir, 'data', 'airfoil_utils'),
        os.path.join(root_dir, 'neural_net')
    ]
    for folder in folders_to_fix:
        if not os.path.exists(folder):
            print(f"Warning: Folder {folder} missing.")
        init_file = os.path.join(folder, '__init__.py')
        if os.path.exists(folder) and not os.path.exists(init_file):
            with open(init_file, 'w') as f: pass

ensure_package_structure()

try:
    from data.airfoil_utils.generate_airfoil_parameterization import fit_catmullrom, get_catmullrom_points
    from neural_net.net_def import NeuralNetwork
except ImportError as e:
    print(f"\nCRITICAL ERROR: {e}")
    print("Ensure you are running this script from the root of your project folder.")
    sys.exit(1)



# --- 2. CONFIGURATION ---
CONFIG = {
    # File Paths
    "MODEL_PATH": "neural_net/trained_nets/FullData_300nodes_10layers_ensemble/xfoil_net_Epoch_2224_Jtrain8.185e-02_Jval_9.928e-01.pth",
    "SEED_AIRFOIL": "naca0006",
    
    # Defaults (Will be overwritten by User Input)
    "MODE": "MAXIMIZE", 
    "TARGET_LD": 75.0,
    
    # GA Settings
    "GENERATIONS": 150,
    "POP_SIZE": 100,      
    "PARENTS_MATING": 20,
    "KEEP_ELITES": 5,
    "BASE_MUTATION_RATE": 15,
    
    # Physics Constraints
    "MIN_THICKNESS": 0.06,         # 6% thickness limit
    "THICKNESS_PENALTY": 100000.0,
    "SMOOTHNESS_PENALTY": 50000.0,
    
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}




# --- 3. RESOURCE LOADING ---
def load_resources():
    print(f"--- Loading Resources ({CONFIG['DEVICE']}) ---")
    
    # 1. Load Neural Network
    model = NeuralNetwork(24, 300, 10)
    if os.path.exists(CONFIG["MODEL_PATH"]):
        checkpoint = torch.load(CONFIG["MODEL_PATH"], map_location=CONFIG["DEVICE"])
        # Handle state dict keys
        state = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint))
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
    else:
        sys.exit(f"Model file not found at: {CONFIG['MODEL_PATH']}")
    
    model.to(CONFIG["DEVICE"]).eval()
    
    # 2. Load Seed Airfoil
    dat_path = f"data/airfoil_database/airfoils/{CONFIG['SEED_AIRFOIL']}.dat"
    if not os.path.exists(dat_path):
        sys.exit(f"Seed airfoil not found at: {dat_path}")
        
    X = np.loadtxt(dat_path)
    X -= np.mean(X, axis=0) # Center it
    fitted = fit_catmullrom(X.flatten(), num_control_pts=12)
    if isinstance(fitted, torch.Tensor): fitted = fitted.detach().cpu().numpy()
    seed_pts = fitted.reshape(-1, 2)
    
    return model, seed_pts

# Load globally
surrogate_model, seed_ctrl_pts = load_resources()





# --- 4. CORE LOGIC (Fitness & Constraints) ---

def get_thickness(ctrl_pts_full):
    """Calculates max thickness ratio of the airfoil."""
    full_tensor = torch.tensor(ctrl_pts_full, dtype=torch.float32).reshape(-1, 2)
    curve_pts = get_catmullrom_points(full_tensor, num_sample_pts=201).detach().cpu().numpy()
    mid = len(curve_pts) // 2
    return np.max(curve_pts[:mid, 1] - curve_pts[-mid:, 1][::-1])

def fitness_func(ga_instance, solution, solution_idx):
    """
    Evaluates fitness based on Mode (Maximize or Target) + Constraints.
    """
    # Reconstruct shape
    interior = np.array(solution).reshape(-1, 2)
    full_shape = np.vstack([seed_ctrl_pts[0:1], interior, seed_ctrl_pts[-1:]])
    
    # 1. AI Prediction
    inp = torch.tensor(full_shape.flatten(), dtype=torch.float32).unsqueeze(0).to(CONFIG["DEVICE"])
    with torch.no_grad():
        pred_ld = surrogate_model(inp).item()
        
    # 2. Constraints (Smoothness & Thickness)
    # Smoothness: Penalize 2nd derivative (jerk/curvature changes)
    curv = np.diff(np.diff(full_shape[:, 1]))
    loss_smooth = np.sum(np.square(curv)) * CONFIG["SMOOTHNESS_PENALTY"]
    
    # Thickness: Penalize if thinner than 6%
    thick = get_thickness(full_shape)
    loss_thick = 0
    if thick < CONFIG["MIN_THICKNESS"]:
        loss_thick = (CONFIG["MIN_THICKNESS"] - thick) * CONFIG["THICKNESS_PENALTY"]
    
    # 3. Mode Selection Logic
    if CONFIG["MODE"] == "MAXIMIZE":
        # Objective: Higher is better
        base_score = pred_ld
    else:
        # Objective: Minimizing Distance to Target (Higher negative error is worse)
        # Error 0 -> Fitness 0. Error 10 -> Fitness -100
        error = abs(pred_ld - CONFIG["TARGET_LD"])
        base_score = -error * 10.0 
        
    # Final Fitness = Objective - Penalties
    return base_score - loss_smooth - loss_thick






# --- 5. ADAPTIVE GA CALLBACK ---
history = {"fitness": [], "diversity": [], "mutation_rate": []}
stagnation_counter = 0
best_fitness_global = -999999

def on_generation(ga_inst):
    """
    Monitors evolution. If stagnant, triggers an 'Adaptive Mutation Boost'.
    """
    global stagnation_counter, best_fitness_global
    
    # 1. Gather Stats
    current_best = ga_inst.best_solution()[1]
    pop_genes = ga_inst.population
    # Diversity = Mean Standard Deviation of genes across population
    diversity = np.mean(np.std(pop_genes, axis=0)) 
    
    history["fitness"].append(current_best)
    history["diversity"].append(diversity)
    history["mutation_rate"].append(ga_inst.mutation_percent_genes)
    
    # 2. Adaptive Logic
    # If we improved significantly, reset stagnation
    if current_best > best_fitness_global + 0.05: 
        best_fitness_global = current_best
        stagnation_counter = 0
        # Cooldown: Return to normal mutation
        ga_inst.mutation_percent_genes = CONFIG["BASE_MUTATION_RATE"]
    else:
        stagnation_counter += 1
        
    # 3. Trigger "Explosion" if stuck
    # If stuck for 15 gens, Triple the mutation rate!
    if stagnation_counter > 15:
        print(f"  [!] Stagnation detected ({stagnation_counter} gens). BOOSTING mutation to 40%!")
        ga_inst.mutation_percent_genes = 40.0 
        stagnation_counter = 0 # Reset to give it time to settle
        
    print(f"Gen {ga_inst.generations_completed:03d} | Fit: {current_best:.2f} | Div: {diversity:.4f} | Mut: {ga_inst.mutation_percent_genes}%")

# --- 6. VISUALIZATION DASHBOARD ---
def plot_dashboard(final_sol, final_ld):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Advanced Evolutionary Dashboard ({CONFIG['MODE']} Mode)", fontsize=16)
    
    # Plot 1: Fitness Convergence
    axs[0, 0].plot(history["fitness"], 'g-', lw=2)
    axs[0, 0].set_title("Fitness Convergence (Learning Curve)")
    axs[0, 0].set_ylabel("Fitness Score")
    axs[0, 0].set_xlabel("Generation")
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Diversity & Mutation (The "Intelligent" Plot)
    ax2 = axs[0, 1]
    ax2.plot(history["diversity"], 'b-', label="Population Diversity")
    ax2.set_ylabel("Gene Std Dev (Diversity)", color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    ax3 = ax2.twinx()
    ax3.plot(history["mutation_rate"], 'r--', alpha=0.5, label="Adaptive Mutation Rate")
    ax3.set_ylabel("Mutation %", color='r')
    ax3.tick_params(axis='y', labelcolor='r')
    axs[0, 1].set_title("Adaptive Dynamics: Diversity vs Mutation")
    
    # Plot 3: Shape Comparison
    seed_curve = get_catmullrom_points(torch.tensor(seed_ctrl_pts, dtype=torch.float32), 201).numpy()
    
    best_interior = np.array(final_sol).reshape(-1, 2)
    best_full = np.vstack([seed_ctrl_pts[0:1], best_interior, seed_ctrl_pts[-1:]])
    best_curve = get_catmullrom_points(torch.tensor(best_full, dtype=torch.float32), 201).numpy()
    
    axs[1, 0].plot(seed_curve[:,0], seed_curve[:,1], 'k--', label='Baseline', alpha=0.5)
    axs[1, 0].plot(best_curve[:,0], best_curve[:,1], 'b-', lw=2, label=f'Evolved (L/D={final_ld:.1f})')
    axs[1, 0].set_title("Airfoil Geometry Optimization")
    axs[1, 0].axis('equal')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Final Population Trade-off
    final_pop = ga_instance.population
    pop_lds = []
    pop_thicks = []
    
    # Re-evaluate final population for scatter plot
    for ind in final_pop:
        interior = np.array(ind).reshape(-1, 2)
        full = np.vstack([seed_ctrl_pts[0:1], interior, seed_ctrl_pts[-1:]])
        t_inp = torch.tensor(full.flatten(), dtype=torch.float32).unsqueeze(0).to(CONFIG["DEVICE"])
        with torch.no_grad():
            pop_lds.append(surrogate_model(t_inp).item())
        pop_thicks.append(get_thickness(full))
        
    sc = axs[1, 1].scatter(pop_thicks, pop_lds, c=pop_lds, cmap='viridis', alpha=0.7)
    axs[1, 1].axvline(CONFIG["MIN_THICKNESS"], color='r', linestyle='--', label='Min Thickness')
    if CONFIG["MODE"] == "TARGET":
        axs[1, 1].axhline(CONFIG["TARGET_LD"], color='orange', linestyle='-', label='Target L/D')
        
    axs[1, 1].set_title("Final Population: Thickness vs L/D")
    axs[1, 1].set_xlabel("Thickness Ratio")
    axs[1, 1].set_ylabel("Predicted L/D")
    axs[1, 1].legend()
    plt.colorbar(sc, ax=axs[1, 1], label="L/D")
    
    plt.tight_layout()
    plt.savefig("Final_Task5_Dashboard.png", dpi=300)
    print("\n✓ Dashboard saved to 'Final_Task5_Dashboard.png'")
    plt.show()

# --- 7. MAIN EXECUTION & USER INPUT ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("   Hybrid Evolutionary Airfoil Optimizer (Task 5)")
    print("="*50)
    print("Select Optimization Mode:")
    print("  [1] MAXIMIZE L/D (Find the absolute best shape)")
    print("  [2] TARGET L/D   (Find a shape with specific L/D)")
    
    user_choice = input("\nEnter choice (1 or 2): ").strip()
    
    if user_choice == "2":
        CONFIG["MODE"] = "TARGET"
        try:
            t_val = float(input("Enter Target L/D Value (e.g., 75.0): ").strip())
            CONFIG["TARGET_LD"] = t_val
        except ValueError:
            print("Invalid number. Defaulting to 75.0")
            CONFIG["TARGET_LD"] = 75.0
        print(f"\n>> MODE SET: TARGET (Aiming for {CONFIG['TARGET_LD']})")
    else:
        CONFIG["MODE"] = "MAXIMIZE"
        print("\n>> MODE SET: MAXIMIZE (Pushing L/D to the limit)")

    # Define Gene Space (Search bounds around seed)
    seed_interior = seed_ctrl_pts[1:-1].flatten()
    init_pop = [seed_interior + np.random.normal(0, 0.005, size=seed_interior.shape) 
                for _ in range(CONFIG["POP_SIZE"])]
    
    # Prevent points from moving too wildly
    gene_space = [{'low': v - 0.08, 'high': v + 0.08} for v in seed_interior]

    print("\n--- Starting Genetic Algorithm ---")
    ga_instance = pygad.GA(
        num_generations=CONFIG["GENERATIONS"],
        num_parents_mating=CONFIG["PARENTS_MATING"],
        fitness_func=fitness_func,
        sol_per_pop=CONFIG["POP_SIZE"],
        num_genes=len(seed_interior),
        initial_population=init_pop,
        gene_space=gene_space,
        mutation_percent_genes=CONFIG["BASE_MUTATION_RATE"],
        keep_parents=CONFIG["KEEP_ELITES"],
        on_generation=on_generation,
        random_seed=42 # For reproducible "wow" results
    )
    
    try:
        ga_instance.run()
        
        # --- RESULTS ---
        best_sol, best_fit, _ = ga_instance.best_solution()
        
        best_interior = np.array(best_sol).reshape(-1, 2)
        best_full = np.vstack([seed_ctrl_pts[0:1], best_interior, seed_ctrl_pts[-1:]])
        
        inp = torch.tensor(best_full.flatten(), dtype=torch.float32).unsqueeze(0).to(CONFIG["DEVICE"])
        with torch.no_grad():
            final_ld = surrogate_model(inp).item()
            
        print("\n" + "="*50)
        print("OPTIMIZATION COMPLETE")
        print("="*50)
        print(f"Mode Used     : {CONFIG['MODE']}")
        print(f"Final L/D     : {final_ld:.4f}")
        if CONFIG["MODE"] == "TARGET":
            print(f"Target L/D    : {CONFIG['TARGET_LD']}")
            print(f"Error         : {abs(final_ld - CONFIG['TARGET_LD']):.4f}")
        print(f"Thickness     : {get_thickness(best_full):.4f}")
        
        plot_dashboard(best_sol, final_ld)
        
    except Exception as e:
        print(f"\nRUNTIME ERROR: {e}")