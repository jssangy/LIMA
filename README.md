# LIMA MAPF

This GitHub is for IEEE RA-L review purposes.

### Prerequisites
- Python 3.9+
- A C++ compiler (GCC, Clang, or MSVC) for building extensions.

### Install Dependencies
Clone the repository and install the required Python packages:

    pip install -r requirements.txt


### Build Extensions
Before running the simulator, build the C++ extensions in-place:

    python setup.py build_ext --inplace

    
### Run Simulation
After installing dependencies, you can run the simulation with:

    python main.py

    
### Argument summary:
```
  --map            Map name under assets/<map>/ (loads assets/<map>/<map>.map)

  --density        Density (%) of agents w.r.t. free tiles (ignored if --num-amrs > 0)

  --num-amrs       If >0, sets the exact number of agents and overrides --density

  --max-steps      Max simulation timesteps before termination

  --planner        Global planner: bfs or cbs

  --workers        Number of workers used by planning/scheduling (implementation-dependent)

  --cache-db-path  Path to schedule cache DB (sqlite)

  --task-mode      random: random tasks, scen: scenario-based tasks from .scen

  --scen-idx       Scenario index for scen mode (loads assets/<map>/scen/<map>_s<idx>.scen)

  --seed           Random seed for reproducibility
```


### Common examples:
Run with density (percentage of free tiles)
  
    python main.py --map cross-30-30 --density 30

Run with a fixed number of agents (overrides --density)
  
    python main.py --map cross-30-30 --num-amrs 200

Run a scenario-based task (s0~s9)
  
    python main.py --map cross-30-30 --task-mode scen --scen-idx 3

Use a different global planner
  
    python main.py --planner cbs
