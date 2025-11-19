#!/usr/bin/env python3
"""
UAV Communication Path Visualization Tool

This script visualizes the UAV communication path formation between User A and User B
in the UAVCommAB environment. It provides dynamic visualization of:
- UAV positions and movements
- Communication links between UAVs
- Formed path from User A to User B
- Performance metrics (delay, throughput, etc.)

Usage:
    python render_route.py --agent-num 6 --steps 100 --save-video

Dependencies:
    matplotlib, numpy
"""

import argparse
import time
import json
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import the environment
from envs.env_core import UAVCommAB

def parse_args():
    parser = argparse.ArgumentParser(description='UAV Communication Path Visualization')
    parser.add_argument('--agent-num', type=int, default=6, help='Number of UAV agents')
    parser.add_argument('--steps', type=int, default=100, help='Number of steps to simulate')
    parser.add_argument('--area-size', type=float, nargs=3, default=[1000.0, 1000.0, 200.0],
                       help='Simulation area size [x, y, z]')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--save-video', action='store_true', help='Save animation as video')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for video')
    parser.add_argument('--output-dir', type=str, default='./uav_path_videos',
                       help='Directory to save videos and logs')
    return parser.parse_args()

def generate_random_actions(env, agent_num):
    """Generate random actions for simulation (for demonstration purposes)."""
    # Random movement with slight bias towards center (simplified policy)
    max_speed = env.max_speed * env.dt
    dx = np.random.uniform(-max_speed, max_speed, agent_num)
    dy = np.random.uniform(-max_speed, max_speed, agent_num)
    # Add slight downward bias to keep UAVs lower
    dz = np.random.uniform(-max_speed * 0.5, max_speed * 0.3, agent_num)
    # Medium power scale
    power = np.random.uniform(0.5, 0.8, agent_num)
    
    actions = np.column_stack([dx, dy, dz, power])
    return actions.tolist()

def get_node_color(node_idx, agent_num):
    """Return appropriate color for different node types."""
    if node_idx == agent_num:  # User A
        return '#ff4444'  # Red
    elif node_idx == agent_num + 1:  # User B
        return '#4444ff'  # Blue
    else:  # UAV
        return '#44ff44'  # Green

def get_node_marker(node_idx, agent_num):
    """Return appropriate marker for different node types."""
    if node_idx == agent_num:  # User A
        return '^'  # Triangle up
    elif node_idx == agent_num + 1:  # User B
        return 'v'  # Triangle down
    else:  # UAV
        return 'o'  # Circle

def get_node_size(node_idx, agent_num):
    """Return appropriate size for different node types."""
    if node_idx >= agent_num:  # User
        return 100
    else:  # UAV
        return 50

def run_simulation(args):
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create environment
    env = UAVCommAB(
        agent_num=args.agent_num,
        area_size=tuple(args.area_size),
        seed=args.seed
    )
    
    # Initialize environment
    obs = env.reset()
    
    # Store simulation data
    sim_data = []
    metrics = {
        'steps': [],
        'delay': [],
        'throughput': [],
        'path_exists': [],
        'energy': []
    }
    
    # Run simulation
    for step in range(args.steps):
        # Generate actions (in a real scenario, these would come from a policy)
        actions = generate_random_actions(env, args.agent_num)
        
        # Take step
        obs, rewards, dones, infos = env.step(actions)
        
        # Record state
        state = env.get_state()
        sim_data.append(state)
        
        # Record metrics
        metrics['steps'].append(step)
        metrics['delay'].append(state['path_delay'])
        metrics['throughput'].append(state['path_throughput'])
        metrics['path_exists'].append(1.0 if state['has_path'] else 0.0)
        metrics['energy'].append(np.mean(state['energy']))
        
        # Print progress
        if step % 10 == 0 or step == args.steps - 1:
            print(f"Step {step}/{args.steps-1}: "
                  f"Path: {'Yes' if state['has_path'] else 'No'}, "
                  f"Delay: {state['path_delay']:.4f}s, "
                  f"Throughput: {state['path_throughput']/1e6:.4f}Mbps, "
                  f"Energy: {np.mean(state['energy']):.2f}")
        
        # Check if episode is done
        if dones[0]:
            print(f"Episode finished at step {step}")
            break
    
    # Save metrics
    metrics_file = output_dir / f"simulation_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_file}")
    
    return sim_data, env

def visualize_simulation(sim_data, env, args):
    """Create visualization of the simulation."""
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # 3D plot for UAV positions and communication paths
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    
    # 2D metrics plots
    ax_metrics = fig.add_subplot(gs[0, 1])
    ax_delay = fig.add_subplot(gs[1, 0])
    ax_energy = fig.add_subplot(gs[1, 1])
    
    # Set up 3D axis limits
    area = env.area
    ax3d.set_xlim(0, area[0])
    ax3d.set_ylim(0, area[1])
    ax3d.set_zlim(0, area[2])
    ax3d.set_xlabel('X Position (m)')
    ax3d.set_ylabel('Y Position (m)')
    ax3d.set_zlabel('Z Position (m)')
    ax3d.set_title('UAV Communication Network')
    
    # Prepare scatter plots for nodes
    uav_pos = sim_data[0]['pos']
    user_a_pos = sim_data[0]['user_a_pos']
    user_b_pos = sim_data[0]['user_b_pos']
    
    # Plot all nodes initially
    uav_scatter = ax3d.scatter(
        uav_pos[:, 0], uav_pos[:, 1], uav_pos[:, 2],
        c='g', marker='o', s=50, alpha=0.8
    )
    
    user_a_scatter = ax3d.scatter(
        [user_a_pos[0]], [user_a_pos[1]], [user_a_pos[2]],
        c='r', marker='^', s=100
    )
    
    user_b_scatter = ax3d.scatter(
        [user_b_pos[0]], [user_b_pos[1]], [user_b_pos[2]],
        c='b', marker='v', s=100
    )
    
    # Create text annotations for labels
    ax3d.text(
        user_a_pos[0], user_a_pos[1], user_a_pos[2],
        'User A', fontsize=10, zorder=10
    )
    
    ax3d.text(
        user_b_pos[0], user_b_pos[1], user_b_pos[2],
        'User B', fontsize=10, zorder=10
    )
    
    # Initialize lists for dynamic elements
    all_link_lines = []
    path_link_lines = []
    status_text = ax3d.text2D(0.02, 0.98, '', transform=ax3d.transAxes)
    
    # Set up metrics plots
    ax_metrics.set_title('Path Status Over Time')
    ax_metrics.set_xlabel('Step')
    ax_metrics.set_ylabel('Path Exists')
    ax_metrics.set_ylim(-0.1, 1.1)
    
    ax_delay.set_title('End-to-End Delay')
    ax_delay.set_xlabel('Step')
    ax_delay.set_ylabel('Delay (s)')
    ax_delay.set_ylim(0, 10)  # Adjust as needed
    
    ax_energy.set_title('Average UAV Energy')
    ax_energy.set_xlabel('Step')
    ax_energy.set_ylabel('Energy')
    ax_energy.set_ylim(0, env.init_energy * 1.1)
    
    # Prepare metrics data
    steps = list(range(len(sim_data)))
    delays = [d['path_delay'] for d in sim_data]
    throughputs = [d['path_throughput'] for d in sim_data]
    path_exists = [1.0 if d['has_path'] else 0.0 for d in sim_data]
    energies = [np.mean(d['energy']) for d in sim_data]
    
    # Plot metrics lines
    path_line, = ax_metrics.plot([], [], 'b-', linewidth=2)
    delay_line, = ax_delay.plot([], [], 'r-', linewidth=2)
    energy_line, = ax_energy.plot([], [], 'g-', linewidth=2)
    
    # Animation update function
    def update(frame):
        nonlocal all_link_lines, path_link_lines
        
        # Get current state
        state = sim_data[frame]
        
        # Update UAV positions
        uav_pos = state['pos']
        uav_scatter._offsets3d = (
            uav_pos[:, 0], uav_pos[:, 1], uav_pos[:, 2]
        )
        
        # Update link lines
        # First remove old lines
        for line in all_link_lines + path_link_lines:
            line.remove()
        all_link_lines = []
        path_link_lines = []
        
        # Plot all possible communication links (faded)
        # First get all nodes including users
        all_nodes = np.zeros((env.agent_num + 2, 3))
        all_nodes[:env.agent_num] = uav_pos
        all_nodes[env.agent_num] = state['user_a_pos']
        all_nodes[env.agent_num + 1] = state['user_b_pos']
        
        # For simplicity, we'll just draw a subset of possible links
        # In a real implementation, you would use the actual graph edges
        for i in range(env.agent_num + 2):
            for j in range(i + 1, env.agent_num + 2):
                # Only draw links for nearby nodes to reduce clutter
                dist = np.linalg.norm(all_nodes[i] - all_nodes[j])
                if dist < 300:  # Threshold for drawing links
                    line = ax3d.plot(
                        [all_nodes[i, 0], all_nodes[j, 0]],
                        [all_nodes[i, 1], all_nodes[j, 1]],
                        [all_nodes[i, 2], all_nodes[j, 2]],
                        'gray', linestyle='-', alpha=0.2, linewidth=1
                    )[0]
                    all_link_lines.append(line)
        
        # Highlight the current A-to-B path if it exists
        if state['has_path'] and len(state['current_path']) > 1:
            for i in range(len(state['current_path']) - 1):
                src = state['current_path'][i]
                dst = state['current_path'][i + 1]
                
                line = ax3d.plot(
                    [all_nodes[src, 0], all_nodes[dst, 0]],
                    [all_nodes[src, 1], all_nodes[dst, 1]],
                    [all_nodes[src, 2], all_nodes[dst, 2]],
                    'orange', linestyle='-', alpha=1.0, linewidth=3
                )[0]
                path_link_lines.append(line)
        
        # Update status text
        status_text.set_text(
            f"Step: {frame}\n" +
            f"Path: {'Yes' if state['has_path'] else 'No'}\n" +
            f"Path Length: {len(state['current_path'])}\n" +
            f"Delay: {state['path_delay']:.4f}s\n" +
            f"Throughput: {state['path_throughput']/1e6:.4f}Mbps"
        )
        
        # Update metrics plots
        path_line.set_data(steps[:frame+1], path_exists[:frame+1])
        delay_line.set_data(steps[:frame+1], delays[:frame+1])
        energy_line.set_data(steps[:frame+1], energies[:frame+1])
        
        # Auto-scale axes if needed
        ax_delay.relim()
        ax_delay.autoscale_view()
        
        # Return all artists that need to be updated
        artists = [uav_scatter, status_text, path_line, delay_line, energy_line]
        artists.extend(all_link_lines)
        artists.extend(path_link_lines)
        
        return artists
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(sim_data),
        interval=100, blit=True
    )
    
    # Save video if requested
    if args.save_video:
        video_file = Path(args.output_dir) / \
                    f"uav_path_animation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        ani.save(str(video_file), fps=args.fps, codec='mpeg4')
        print(f"Animation saved to {video_file}")
    
    # Show plot
    plt.tight_layout()
    plt.show()

def main():
    args = parse_args()
    print(f"Starting UAV Communication Path Visualization")
    print(f"Parameters: {vars(args)}")
    
    # Run simulation
    print("Running simulation...")
    sim_data, env = run_simulation(args)
    
    # Visualize results
    print("Generating visualization...")
    visualize_simulation(sim_data, env, args)

if __name__ == "__main__":
    import sys
    main()