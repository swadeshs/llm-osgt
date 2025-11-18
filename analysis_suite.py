#%%
import os
import re
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import ast
import scipy.optimize
import openai
import traceback
import statsmodels.api as sm
from dotenv import load_dotenv
from typing import Optional, List, Dict, Tuple, Any, Callable
from collections import Counter

# Code complexity imports
from radon.raw import analyze as analyze_raw
from radon.metrics import mi_visit, h_visit
from radon.complexity import cc_visit

import importlib.util

try:
    import evaluation_harness as harness
except ImportError:
    print("Error: Could not import evaluation_harness.py. Make sure it's in the same directory.")

#%%
# --- Matplotlib Configuration ---
plt.rcParams.update({'font.size': 16, 
                     'figure.autolayout': True, 
                     'font.family': 'monospace'})
AXIS_LABEL_FONTSIZE = 18
TITLE_FONTSIZE = 18
LEGEND_FONTSIZE = 16
OPPONENT_CODE_VARIABLE_NAME = "opponent_program_code" 
# --- LLM-as-Judge Configuration ---
LLM_JUDGE_MODEL = "gpt-4o"
STRATEGIC_RESPONSE_CATEGORIES = [
    "Independent_Development", "Counter_Measure", "Exploitation_Attempt",
    "Direct_Imitation", "Feint", "Data_Missing", "Error"
]
STRATEGY_PLOT_COLORS = {
    "Independent_Development":  "#FF8C00",
    "Counter_Measure": "#006400",
    "Exploitation_Attempt": "#8B0000",
    "Direct_Imitation": "#4682B4",
    "Feint": "#9932CC",
    "Data_Missing": "#A9A9A9",
    "Error": "#000000",
    "default": "#708090"
}

def load_api_key():
    """Loads OpenAI API key from .env file."""
    try:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found in .env file or environment.")
            return None
        openai.api_key = api_key
        print("OpenAI API Key configured successfully.")
        return openai.OpenAI()
    except Exception as e:
        print(f"Error loading OpenAI API key: {e}")
        return None

# --- Data Loading & Processing ---

def find_experiment_runs(base_dir: Path) -> List[Path]:
    """Finds all individual run directories within the base experiment folder."""
    run_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('seed_')]
    print(f"Found {len(run_dirs)} experiment run(s) in '{base_dir}'.")
    return run_dirs

def load_file_content(filepath: Path) -> Optional[str]:
    """Safely loads text content from a file."""
    if filepath and filepath.exists():
        try:
            return filepath.read_text(encoding='utf-8')
        except Exception as e:
            print(f"Warning: Could not read file {filepath}: {e}")
    return None

def get_radon_metrics(code: Optional[str]) -> Dict[str, Any]:
    """Calculates all relevant Radon metrics for a piece of code."""
    zero_metrics = {
        'sloc': 0, 'cyclomatic_complexity': 0, 'halstead_effort': 0,
        'halstead_volume': 0, 'halstead_difficulty': 0,
        'halstead_time': 0, 'halstead_bugs': 0
    }
    if not code:
        return zero_metrics

    try:
        raw = analyze_raw(code)
        cc_blocks = cc_visit(code)
        avg_cc = np.mean([block.complexity for block in cc_blocks]) if cc_blocks else 0
        halstead = h_visit(code).total

        return {
            'sloc': raw.sloc,
            'cyclomatic_complexity': avg_cc,
            'halstead_effort': halstead.effort,
            'halstead_volume': halstead.volume,
            'halstead_difficulty': halstead.difficulty,
            'halstead_time': halstead.time,
            'halstead_bugs': halstead.bugs
        }
    except SyntaxError as e:
        print(f"Warning: Radon could not parse code, returning zero-metrics. Error: {e}")
        return zero_metrics

def average_radon_metrics_across_experiments(results_base_dir: Path, games_to_run: Optional[List[str]] = None) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Finds all experiments (pairing folders), aggregates metrics, and returns the mean, SEM, and per-experiment data.
    Can be filtered to run only on specific games.
    """
    print(f"\n{'='*20} AGGREGATING METRICS ACROSS ALL EXPERIMENTS {'='*20}")
    print(f"Scanning for experiments in: {results_base_dir}")

    top_level_dirs = [d for d in results_base_dir.iterdir() if d.is_dir() and d.name.startswith('results_')]
    
    if games_to_run:
        print(f"Filtering for games: {games_to_run}")
        top_level_dirs = [d for d in top_level_dirs if any(game_name in d.name for game_name in games_to_run)]

    all_experiment_dirs = []
    for top_dir in top_level_dirs:
        pairing_dirs = [d for d in top_dir.iterdir() if d.is_dir() and not d.name.endswith('_plots')]
        all_experiment_dirs.extend(pairing_dirs)

    if not all_experiment_dirs:
        print("Error: No experiment pairing folders found to aggregate.")
        return None, None, None
    
    print(f"Found {len(all_experiment_dirs)} experiment pairings to process.")

    all_runs_data = []
    for experiment_dir in all_experiment_dirs:
        descriptive_name = f"{experiment_dir.parent.name}/{experiment_dir.name}"
        print(f"  - Processing {descriptive_name}...")
        
        run_dirs = find_experiment_runs(experiment_dir)
        for run_dir in run_dirs:
            processed_data = process_dyadic_run(run_dir)
            if processed_data and processed_data.get('type') == 'dyadic':
                processed_data['data']['experiment'] = descriptive_name
                all_runs_data.append(processed_data['data'])

    if not all_runs_data:
        print("\nNo valid dyadic run data found across all experiments.")
        return None, None, None

    combined_df = pd.concat(all_runs_data, ignore_index=True)
    print(f"\nAggregated data from a total of {len(all_runs_data)} runs.")

    metrics_to_average = [
        'sloc', 'cyclomatic_complexity', 'halstead_effort', 'halstead_volume',
        'halstead_difficulty', 'halstead_time', 'halstead_bugs', 'osas'
    ]
    for metric in metrics_to_average:
        combined_df[metric] = combined_df[[f'{metric}_A', f'{metric}_B']].mean(axis=1)

    per_experiment_mean_df = combined_df.groupby(['experiment', 'meta_round'])[metrics_to_average].mean()
    grouped = combined_df.groupby('meta_round')
    mean_df = grouped[metrics_to_average].mean()
    sem_df = grouped[metrics_to_average].sem()

    return mean_df, sem_df, per_experiment_mean_df

def average_payoffs_across_experiments(results_base_dir: Path, games_to_run: Optional[List[str]] = None) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Finds all experiments, aggregates logged scores, and returns mean, SEM, and per-experiment data.
    """
    print(f"\n{'='*20} AGGREGATING PAYOFFS ACROSS ALL EXPERIMENTS {'='*20}")
    print(f"Scanning for experiments in: {results_base_dir}")

    top_level_dirs = [d for d in results_base_dir.iterdir() if d.is_dir() and d.name.startswith('results_')]
    
    if games_to_run:
        print(f"Filtering for games: {games_to_run}")
        top_level_dirs = [d for d in top_level_dirs if any(game_name in d.name for game_name in games_to_run)]

    all_experiment_dirs = []
    for top_dir in top_level_dirs:
        pairing_dirs = [d for d in top_dir.iterdir() if d.is_dir() and not d.name.endswith('_plots')]
        all_experiment_dirs.extend(pairing_dirs)

    if not all_experiment_dirs:
        print("Error: No experiment pairing folders found to aggregate.")
        return None, None, None
    
    print(f"Found {len(all_experiment_dirs)} experiment pairings to process.")

    all_runs_data = []
    for experiment_dir in all_experiment_dirs:
        descriptive_name = f"{experiment_dir.parent.name}/{experiment_dir.name}"
        
        run_dirs = find_experiment_runs(experiment_dir)
        for run_dir in run_dirs:
            log_files = list(run_dir.glob("log_seed_*.csv"))
            if not log_files:
                continue
            
            try:
                log_df = pd.read_csv(log_files[0])
                log_df['experiment'] = descriptive_name
                log_df['payoff'] = log_df[['score_A', 'score_B']].mean(axis=1)
                all_runs_data.append(log_df[['experiment', 'meta_round', 'payoff']])
            except Exception as e:
                print(f"Warning: Could not process log file in {run_dir}: {e}")

    if not all_runs_data:
        print("\nNo valid log data found across all experiments.")
        return None, None, None

    combined_df = pd.concat(all_runs_data, ignore_index=True)
    print(f"\nAggregated payoff data from a total of {len(all_runs_data)} runs.")

    per_experiment_mean_df = combined_df.groupby(['experiment', 'meta_round'])[['payoff']].mean()
    grouped = combined_df.groupby('meta_round')
    mean_df = grouped[['payoff']].mean()
    sem_df = grouped[['payoff']].sem()

    return mean_df, sem_df, per_experiment_mean_df

class TaintTracker(ast.NodeVisitor):
    """
    An AST visitor to track taint from a source variable and score its usage.
    """
    def __init__(self, source_variable_name: str):
        self.source_name = source_variable_name
        self.tainted_vars: Set[str] = {source_variable_name}
        self.score = 0
        self.scored_nodes: Set[ast.AST] = set()

    def _get_node_id(self, node: ast.AST) -> Optional[str]:
        """Gets the string identifier for a variable node."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            # Represents 'self.opponent_code' as one identifier
            base = self._get_node_id(node.value)
            if base:
                return f"{base}.{node.attr}"
        return None

    def is_tainted(self, node: ast.AST) -> bool:
        """Recursively checks if a node or its children are tainted."""
        node_id = self._get_node_id(node)
        if node_id and node_id in self.tainted_vars:
            return True
        for child in ast.iter_child_nodes(node):
            if self.is_tainted(child):
                return True
        return False

    def visit_Assign(self, node: ast.Assign):
        """Tracks taint propagation through assignments."""
        if self.is_tainted(node.value):
            for target in node.targets:
                target_id = self._get_node_id(target)
                if target_id:
                    self.tainted_vars.add(target_id)
        else: # Handle untainting on reassignment
            for target in node.targets:
                target_id = self._get_node_id(target)
                if target_id and target_id in self.tainted_vars:
                    self.tainted_vars.remove(target_id)
        self.generic_visit(node)

    def _increment_score(self, node: ast.AST):
        """Increments score for a node if it hasn't been scored yet."""
        if node not in self.scored_nodes:
            # Find the parent statement to score it only once per line
            current = node
            while hasattr(current, 'parent') and not isinstance(current, ast.stmt):
                current = current.parent
            if current not in self.scored_nodes:
                self.score += 1
                self.scored_nodes.add(current)


    def visit_Call(self, node: ast.Call):
        """Scores usage of tainted variables in function calls."""
        if any(self.is_tainted(arg) for arg in node.args) or \
           any(self.is_tainted(kw.value) for kw in node.keywords) or \
           self.is_tainted(node.func):
            self._increment_score(node)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp):
        """Scores usage in binary operations."""
        if self.is_tainted(node.left) or self.is_tainted(node.right):
            self._increment_score(node)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare):
        """Scores usage in comparisons."""
        if self.is_tainted(node.left) or any(self.is_tainted(comp) for comp in node.comparators):
            self._increment_score(node)
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        """Scores usage in for loops."""
        if self.is_tainted(node.iter):
            self._increment_score(node)
        self.generic_visit(node)

def calculate_osas(code: Optional[str], opponent_var_name: str) -> int:
    """
    Calculates the Opponent Script Access Score (OSAS) for a given code string.
    """
    if not code:
        return 0
    try:
        tree = ast.parse(code)
        # Add parent pointers for context
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node
        tracker = TaintTracker(opponent_var_name)
        tracker.visit(tree)
        return tracker.score
    except SyntaxError as e:
        print(f"Warning: OSAS could not parse code, returning 0. Error: {e}")
        return 0

def extract_pairing_name(folder_name: str) -> str:
    """
    Extracts a clean 'AGENT vs AGENT' legend label from a long folder name.
    Example: 'results_CoinGame/Dyadic_CoinGame_CPM_vs_DPM_Kimi-K2' -> 'CPM vs DPM'
    """
    match = re.search(r'(CPM|DPM|PM)_vs_(CPM|DPM|PM)', folder_name)
    if match:
        return match.group(0).replace('_', ' ')
    
    try:
        return folder_name.split('/')[-1].split('_')[2]
    except IndexError:
        return folder_name.split('/')[-1]
    
def plot_aggregated_metrics_panel(mean_df: pd.DataFrame, sem_df: pd.DataFrame, per_experiment_data: Optional[pd.DataFrame], output_dir: Path):
    """
    Generates and saves a single panel plot comparing metrics.
    """
    print("\nGenerating aggregated cross-game panel plot...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    game_map = {
        'CoinGame': {'img_path': 'images/coingame.png', 'row': 0, 'title': 'Coin Game'},
        'IPD': {'img_path': 'images/ipd.png', 'row': 1, 'title': 'IPD'}
    }
    metrics_to_plot = ['cyclomatic_complexity', 'halstead_effort', 'osas']
    metric_labels = {'cyclomatic_complexity': 'Cyclomatic Complexity', 'halstead_effort': 'Halstead Effort', 'osas': 'OSAS'}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

    unique_experiments = per_experiment_data.index.get_level_values('experiment').unique()
    label_map = {name: extract_pairing_name(name) for name in unique_experiments}
    unique_labels = sorted(list(set(label_map.values())))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    label_color_map = dict(zip(unique_labels, colors))
    experiment_color_map = {exp_name: label_color_map[clean_label] for exp_name, clean_label in label_map.items()}

    for game_name, info in game_map.items():
        row = info['row']
        
        for col, metric_key in enumerate(metrics_to_plot, 0):
            ax = axes[row, col]
            
            game_data = per_experiment_data[per_experiment_data.index.get_level_values('experiment').str.contains(game_name)]
            
            if game_data.empty:
                if col == 1:
                    ax.set_title(info['title'], fontsize=TITLE_FONTSIZE)
                ax.text(0.5, 0.5, 'No data for this game', ha='center', va='center')
                continue

            for experiment_name, group in game_data.groupby(level='experiment'):
                exp_metric_data = group[metric_key]
                ax.plot(exp_metric_data.index.get_level_values('meta_round'), exp_metric_data.values,
                        marker='.', linestyle='-', color=experiment_color_map.get(experiment_name, 'gray'), 
                        alpha=0.4, zorder=1)

            game_mean_df = game_data.groupby('meta_round')[metric_key].mean()
            game_sem_df = game_data.groupby('meta_round')[metric_key].sem()

            ax.plot(game_mean_df.index, game_mean_df.values, marker='o', linestyle='-', 
                    color='black', label='Game Average', zorder=2, markersize=5)
            ax.fill_between(game_mean_df.index, game_mean_df.values - game_sem_df.values, game_mean_df.values + game_sem_df.values,
                            color='black', alpha=0.2, zorder=2)
            
            if col == 1:
               ax.set_title(info['title'], fontsize=TITLE_FONTSIZE)
            
            if row == 1:
                ax.set_xlabel("Meta-Round")

            ax.set_ylabel(metric_labels[metric_key])
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    legend_handles = [plt.Line2D([0], [0], color=color, lw=2) for label, color in sorted(label_color_map.items())]
    fig.legend(handles=legend_handles, labels=sorted(label_color_map.keys()), 
               loc='center right', bbox_to_anchor=(1.0, 0.5), title="Pairings")

    plt.tight_layout(rect=[0.05, 0, 0.95, 1.0])
    panel_path = output_dir / "aggregated_metrics_panel.pdf"
    plt.savefig(panel_path)
    print(f"Saved aggregated panel plot to {panel_path}")
    plt.close(fig)

def generate_individual_metric_plots(mean_df: pd.DataFrame, sem_df: pd.DataFrame, per_experiment_data: Optional[pd.DataFrame], output_dir: Path):
    """
    Generates and saves individual plots for each aggregated metric for a specific game.
    """
    print(f"Generating individual metric plots in {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    game_name = output_dir.name.replace('_aggregated_metrics', '')

    plot_info = {
        'sloc': ('Source Lines of Code (SLOC)', 'Average SLOC'),
        'cyclomatic_complexity': ('Cyclomatic Complexity', 'Average Complexity'),
        'halstead_effort': ('Halstead Effort', 'Average Effort'),
        'halstead_volume': ('Halstead Volume', 'Average Volume'),
        'halstead_difficulty': ('Halstead Difficulty', 'Average Difficulty'),
        'halstead_time': ('Halstead Time', 'Average Time (seconds)'),
        'halstead_bugs': ('Halstead Bugs Estimate', 'Average Estimated Bugs'),
        'osas': ('Opponent Script Access Score (OSAS)', 'Average OSAS'),
    }

    # Color mapping needs to be recreated here for the specific subset of data
    unique_experiments = per_experiment_data.index.get_level_values('experiment').unique()
    label_map = {name: extract_pairing_name(name) for name in unique_experiments}
    unique_labels = sorted(list(set(label_map.values())))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    label_color_map = dict(zip(unique_labels, colors))
    experiment_color_map = {exp_name: label_color_map[clean_label] for exp_name, clean_label in label_map.items()}

    for metric_key, (title, ylabel) in plot_info.items():
        if mean_df.empty or metric_key not in mean_df.columns:
            continue
        
        plt.figure(figsize=(10, 6))
        
        y = mean_df[metric_key]
        X = sm.add_constant(mean_df.index)
        model = sm.OLS(y, X).fit()
        p_value = model.pvalues[1]

        for experiment_name, experiment_df in per_experiment_data.groupby(level='experiment'):
            exp_metric_data = experiment_df[metric_key]
            plt.plot(exp_metric_data.index.get_level_values('meta_round'), exp_metric_data.values,
                     marker='.', linestyle='-', color=experiment_color_map.get(experiment_name, 'gray'), alpha=0.3, zorder=1)

        mean_vals = mean_df[metric_key]
        sem_vals = sem_df[metric_key]
        plt.plot(mean_df.index, mean_vals, marker='o', linestyle='-', label='Game-wide Average', color='black', zorder=2)
        plt.fill_between(mean_df.index, mean_vals - sem_vals, mean_vals + sem_vals, color='black', alpha=0.2, zorder=2)
        
        plt.text(0.95, 0.95, f'p = {p_value:.3f}', transform=plt.gca().transAxes,
                 fontsize=14, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        specific_title = f"{title} for {game_name}"
        plt.title(specific_title, fontsize=TITLE_FONTSIZE)
        plt.xlabel("Meta-Round", fontsize=AXIS_LABEL_FONTSIZE)
        plt.ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
        plt.xticks(np.arange(min(mean_df.index), max(mean_df.index) + 1, 1))
        
        legend_handles = [plt.Line2D([0], [0], color=color, lw=2) for label, color in sorted(label_color_map.items())]
        plt.legend(handles=legend_handles, labels=sorted(label_color_map.keys()), title="Pairings", bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.grid(True, linestyle='--', alpha=0.6)
        
        plot_path = output_dir / f"{metric_key}_over_time.pdf"
        plt.savefig(plot_path, bbox_inches='tight')
        print(f"  - Saved plot: {plot_path.name}")
        plt.close()

def process_dyadic_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Processes data from a single dyadic meta-game run."""
    log_files = list(run_dir.glob("log_seed_*.csv"))
    if not log_files:
        return None

    log_path = log_files[0]
    if len(log_files) > 1:
        print(f"Warning: Found multiple log files in {run_dir}. Using the first one: {log_path}")

    try:
        df = pd.read_csv(log_path)
        metrics = []
        for _, row in df.iterrows():
            # Get paths for program code
            prog_A_path = run_dir / "program_strategies" / f"{row['prog_A']}.py"
            prog_B_path = run_dir / "program_strategies" / f"{row['prog_B']}.py"

            agent_A_label = row['agent_A']
            agent_B_label = row['agent_B']
            meta_round = row['meta_round']
            text_A_path = run_dir / "textual_strategies" / f"{agent_A_label}_MR{meta_round}_textual_strategy.txt"
            text_B_path = run_dir / "textual_strategies" / f"{agent_B_label}_MR{meta_round}_textual_strategy.txt"

            code_A = load_file_content(prog_A_path)
            code_B = load_file_content(prog_B_path)

            radon_metrics_A = get_radon_metrics(code_A)
            radon_metrics_B = get_radon_metrics(code_B)

            osas_A = calculate_osas(code_A, OPPONENT_CODE_VARIABLE_NAME)
            osas_B = calculate_osas(code_B, OPPONENT_CODE_VARIABLE_NAME)

            metrics.append({
                'meta_round': row['meta_round'],
                'agent_A': agent_A_label,
                'agent_B': agent_B_label,
                **{f'{k}_A': v for k, v in radon_metrics_A.items()},
                **{f'{k}_B': v for k, v in radon_metrics_B.items()},
                'osas_A': osas_A,
                'osas_B': osas_B,
                'code_A_path': str(prog_A_path),
                'code_B_path': str(prog_B_path),
                'text_A_path': str(text_A_path),
                'text_B_path': str(text_B_path),
            })
        return {'type': 'dyadic', 'data': pd.DataFrame(metrics)}
    except Exception as e:
        print(f"Error processing dyadic run {run_dir.name}: {e}")
        return None

        return None

def process_evolutionary_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Processes data from a single evolutionary tournament run."""
    payoff_path = run_dir / "payoff_matrix.csv"
    moran_path = run_dir / "moran_process_history.csv"
    if not payoff_path.exists():
        return None

    try:
        payoff_matrix = pd.read_csv(payoff_path, index_col=0)
        moran_history = pd.read_csv(moran_path) if moran_path.exists() else None
        return {'type': 'evolutionary', 'payoff_matrix': payoff_matrix, 'moran_history': moran_history}
    except Exception as e:
        print(f"Error processing evolutionary run {run_dir.name}: {e}")
        return None

# --- Dyadic Analysis ---

def analyze_dyadic_metrics(all_runs_data: List[pd.DataFrame], output_dir: Path):
    """Analyzes and plots code metrics for dyadic games."""
    if not all_runs_data:
        print("No dyadic data to analyze.")
        return

    combined_df = pd.concat(all_runs_data, ignore_index=True)
    grouped = combined_df.groupby('meta_round')
    numeric_cols = combined_df.select_dtypes(include=np.number).columns.tolist()
    mean_df = grouped[numeric_cols].mean()
    sem_df = grouped[numeric_cols].sem()

    def plot_metric(metric_key: str, title: str, ylabel: str):
        plt.figure(figsize=(10, 6))
        if f'{metric_key}_A' not in mean_df.columns or f'{metric_key}_B' not in mean_df.columns:
            print(f"Warning: Metric key '{metric_key}' not found in aggregated data. Skipping plot.")
            plt.close()
            return

        for agent_id in ['A', 'B']:
            mean_vals = mean_df[f'{metric_key}_{agent_id}']
            sem_vals = sem_df[f'{metric_key}_{agent_id}']
            agent_label = combined_df[f'agent_{agent_id}'].iloc[0]

            plt.plot(mean_df.index, mean_vals, marker='o', linestyle='-', label=f'Agent {agent_id} ({agent_label})')
            plt.fill_between(mean_df.index, mean_vals - sem_vals, mean_vals + sem_vals, alpha=0.2)

        plt.title(title, fontsize=TITLE_FONTSIZE)
        plt.xlabel("Meta-Round", fontsize=AXIS_LABEL_FONTSIZE)
        plt.ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
        plt.xticks(np.arange(min(mean_df.index), max(mean_df.index)+1, 1))
        plt.legend(fontsize=LEGEND_FONTSIZE)
        plt.grid(True, linestyle='--', alpha=0.6)

        plot_path = output_dir / f"dyadic_{metric_key}_over_time.pdf"
        plt.savefig(plot_path)
        print(f"Saved plot to {plot_path}")
        plt.close()

    plot_metric('sloc', 'Average Source Lines of Code (SLOC) Over Time', 'SLOC')
    plot_metric('cyclomatic_complexity', 'Average Cyclomatic Complexity Over Time', 'Complexity Score')
    plot_metric('halstead_effort', 'Average Halstead Effort Over Time', 'Effort')
    plot_metric('halstead_volume', 'Average Halstead Volume Over Time', 'Volume')
    plot_metric('halstead_difficulty', 'Average Halstead Difficulty Over Time', 'Difficulty')
    plot_metric('halstead_time', 'Average Halstead Time Over Time', 'Time (seconds)')
    plot_metric('halstead_bugs', 'Average Halstead Bugs Estimate Over Time', 'Estimated Bugs')
    plot_metric('osas', 'Opponent Script Access Score (OSAS) Over Time', 'OSAS')

def get_agent_color(agent_name: str) -> str:
    """Returns a specific color based on keywords in the agent's name."""
    if "CPM" in agent_name:
        return 'green'
    elif "DPM" in agent_name:
        return 'red'
    elif "PM" in agent_name:
        return 'black'
    return 'blue'

def _plot_llm_judge_custom_panel(results_df: pd.DataFrame, all_runs_data: List[pd.DataFrame], output_dir: Path):
    """Generates a custom 2x2 panel of line plots for specific strategic responses."""
    agent_A_label = all_runs_data[0]['agent_A'].iloc[0]
    agent_B_label = all_runs_data[0]['agent_B'].iloc[0]
    agent_labels = {'A': agent_A_label, 'B': agent_B_label}

    one_hot = pd.get_dummies(results_df['classification'])
    one_hot = one_hot.reindex(columns=STRATEGIC_RESPONSE_CATEGORIES, fill_value=0)
    results_with_one_hot = pd.concat([results_df, one_hot], axis=1)

    categories_to_plot = ["Counter_Measure", "Exploitation_Attempt", "Direct_Imitation", "Feint"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    axes = axes.flatten()
    fig.suptitle("Evolution of Key Strategic Responses", fontsize=TITLE_FONTSIZE, y=1.0)

    for agent_id, agent_name in agent_labels.items():
        agent_data = results_with_one_hot[results_with_one_hot['agent'] == agent_id]
        grouped = agent_data.groupby('meta_round')
        mean_props = grouped[categories_to_plot].mean()
        sem_props = grouped[categories_to_plot].sem()

        for i, cat in enumerate(categories_to_plot):
            ax = axes[i]
            if cat in mean_props.columns:
                mean_vals = mean_props[cat]
                sem_vals = sem_props[cat].fillna(0)
                line_color = get_agent_color(agent_name)
                
                ax.plot(mean_vals.index, mean_vals, marker='o', linestyle='-', label=agent_name, color=line_color)
                ax.fill_between(mean_vals.index, mean_vals - sem_vals, mean_vals + sem_vals, alpha=0.2, color=line_color)

    max_round = results_df['meta_round'].max()
    for i, ax in enumerate(axes):
        if i < len(categories_to_plot):
            cat = categories_to_plot[i]
            ax.set_title(cat.replace("_", " "), fontsize=LEGEND_FONTSIZE)
            ax.set_xlabel("Meta-Round", fontsize=AXIS_LABEL_FONTSIZE - 2)
            ax.set_xticks(range(1, max_round + 1))
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.set_ylim(-0.05, 1.05)
            if i % 2 == 0:
                ax.set_ylabel("Proportion", fontsize=AXIS_LABEL_FONTSIZE - 2)
        else:
            ax.axis('off')
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=len(agent_labels), fontsize=LEGEND_FONTSIZE)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_path = output_dir / "dyadic_llm_judge_custom_panel_plot.pdf"
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Saved custom panel plot to {plot_path}")
    plt.close(fig)

def _plot_llm_judge_line_plots(results_df: pd.DataFrame, all_runs_data: List[pd.DataFrame], output_dir: Path):
    """Generates detailed line plots for strategic response proportions with updated colors."""
    agent_A_label = all_runs_data[0]['agent_A'].iloc[0]
    agent_B_label = all_runs_data[0]['agent_B'].iloc[0]
    agent_labels = {'A': agent_A_label, 'B': agent_B_label}

    one_hot = pd.get_dummies(results_df['classification'])
    one_hot = one_hot.reindex(columns=STRATEGIC_RESPONSE_CATEGORIES, fill_value=0)
    results_with_one_hot = pd.concat([results_df, one_hot], axis=1)

    categories_to_plot = ["Independent_Development", "Counter_Measure", "Exploitation_Attempt"]

    fig, axes = plt.subplots(1, len(categories_to_plot), figsize=(5.5 * len(categories_to_plot), 5), sharey=True)
    if len(categories_to_plot) == 1: axes = [axes]
    fig.suptitle("Evolution of Strategic Responses", fontsize=TITLE_FONTSIZE, y=1.02)

    for agent_id, agent_name in agent_labels.items():
        agent_data = results_with_one_hot[results_with_one_hot['agent'] == agent_id]
        grouped = agent_data.groupby('meta_round')
        mean_props = grouped[STRATEGIC_RESPONSE_CATEGORIES].mean()
        sem_props = grouped[STRATEGIC_RESPONSE_CATEGORIES].sem()

        for i, cat in enumerate(categories_to_plot):
            ax = axes[i]
            if cat in mean_props.columns:
                mean_vals = mean_props[cat]
                sem_vals = sem_props[cat].fillna(0)
                line_color = get_agent_color(agent_name)
                
                ax.plot(mean_vals.index, mean_vals, marker='o', linestyle='-', label=agent_name, color=line_color)
                ax.fill_between(mean_vals.index, mean_vals - sem_vals, mean_vals + sem_vals, alpha=0.2, color=line_color)

    max_round = results_df['meta_round'].max()
    for i, (ax, cat) in enumerate(zip(axes, categories_to_plot)):
        ax.set_title(cat.replace("_", " "), fontsize=LEGEND_FONTSIZE)
        ax.set_xlabel("Meta-Round", fontsize=AXIS_LABEL_FONTSIZE - 2)
        ax.set_xticks(range(1, max_round + 1))
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.set_ylim(-0.05, 1.05)
        if i == 0:
            ax.set_ylabel("Proportion", fontsize=AXIS_LABEL_FONTSIZE - 2)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=LEGEND_FONTSIZE)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = output_dir / "dyadic_llm_judge_detailed_line_plots.pdf"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)

def _plot_llm_judge_full_panel(results_df: pd.DataFrame, all_runs_data: List[pd.DataFrame], output_dir: Path):
    """Generates a full panel of line plots with updated colors."""
    agent_A_label = all_runs_data[0]['agent_A'].iloc[0]
    agent_B_label = all_runs_data[0]['agent_B'].iloc[0]
    agent_labels = {'A': agent_A_label, 'B': agent_B_label}

    one_hot = pd.get_dummies(results_df['classification'])
    one_hot = one_hot.reindex(columns=STRATEGIC_RESPONSE_CATEGORIES, fill_value=0)
    results_with_one_hot = pd.concat([results_df, one_hot], axis=1)

    categories_to_plot = STRATEGIC_RESPONSE_CATEGORIES

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey=True)
    axes = axes.flatten()
    fig.suptitle("Evolution of All Strategic Responses", fontsize=TITLE_FONTSIZE, y=1.0)

    for agent_id, agent_name in agent_labels.items():
        agent_data = results_with_one_hot[results_with_one_hot['agent'] == agent_id]
        grouped = agent_data.groupby('meta_round')
        mean_props = grouped[categories_to_plot].mean()
        sem_props = grouped[categories_to_plot].sem()

        for i, cat in enumerate(categories_to_plot):
            ax = axes[i]
            if cat in mean_props.columns:
                mean_vals = mean_props[cat]
                sem_vals = sem_props[cat].fillna(0)
                line_color = get_agent_color(agent_name)
                
                ax.plot(mean_vals.index, mean_vals, marker='o', linestyle='-', label=agent_name, color=line_color)
                ax.fill_between(mean_vals.index, mean_vals - sem_vals, mean_vals + sem_vals, alpha=0.2, color=line_color)

    max_round = results_df['meta_round'].max()
    for i, ax in enumerate(axes):
        if i < len(categories_to_plot):
            cat = categories_to_plot[i]
            ax.set_title(cat.replace("_", " "), fontsize=LEGEND_FONTSIZE)
            ax.set_xlabel("Meta-Round", fontsize=AXIS_LABEL_FONTSIZE - 2)
            ax.set_xticks(range(1, max_round + 1))
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.set_ylim(-0.05, 1.05)
            if i % 4 == 0:
                ax.set_ylabel("Proportion", fontsize=AXIS_LABEL_FONTSIZE - 2)
        else:
            ax.axis('off')
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=LEGEND_FONTSIZE)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_path = output_dir / "dyadic_llm_judge_full_panel_plot.pdf"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)

# --- Add this new helper function somewhere in your script ---

def _get_agent_types_from_pairing(pairing_name: str) -> Optional[Tuple[str, str]]:
    """Extracts agent types like ('CPM', 'DPM') from a pairing directory name."""
    match = re.search(r'(CPM|DPM|PM)_vs_(CPM|DPM|PM)', pairing_name)
    if match:
        return match.group(1), match.group(2)
    return None


def load_all_payoff_data_by_agent(results_base_dir: Path, games_to_run: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """
    Finds all experiments, aggregates logged scores BY AGENT TYPE, and returns
    a single long-form DataFrame.
    """
    print(f"\n{'='*20} AGGREGATING PAYOFFS BY AGENT TYPE {'='*20}")
    print(f"Scanning for experiments in: {results_base_dir}")

    top_level_dirs = [d for d in results_base_dir.iterdir() if d.is_dir() and d.name.startswith('results_')]
    
    if games_to_run:
        print(f"Filtering for games: {games_to_run}")
        top_level_dirs = [d for d in top_level_dirs if any(game_name in d.name for game_name in games_to_run)]

    all_experiment_dirs = []
    for top_dir in top_level_dirs:
        pairing_dirs = [d for d in top_dir.iterdir() if d.is_dir() and not d.name.endswith('_plots')]
        all_experiment_dirs.extend(pairing_dirs)

    if not all_experiment_dirs:
        print("Error: No experiment pairing folders found to aggregate.")
        return None
    
    print(f"Found {len(all_experiment_dirs)} experiment pairings to process.")

    all_payoffs_list = []
    for experiment_dir in all_experiment_dirs:
        descriptive_name = f"{experiment_dir.parent.name}/{experiment_dir.name}"
        
        # Get agent types
        agent_types = _get_agent_types_from_pairing(experiment_dir.name)
        if not agent_types:
            print(f"Warning: Could not determine agent types for '{experiment_dir.name}'. Skipping.")
            continue
        agent_A_type, agent_B_type = agent_types
        game_name = experiment_dir.parent.name.replace('results_', '')
        
        run_dirs = find_experiment_runs(experiment_dir)
        for run_dir in run_dirs:
            log_files = list(run_dir.glob("log_seed_*.csv"))
            if not log_files:
                continue
            
            try:
                seed = int(run_dir.name.split('_')[-1])
                log_df = pd.read_csv(log_files[0])
                for _, row in log_df.iterrows():
                    # Add entry for Agent A's score
                    all_payoffs_list.append({
                        'game': game_name,
                        'pairing': descriptive_name,
                        'seed': seed,
                        'meta_round': row['meta_round'],
                        'acting_agent': agent_A_type,
                        'opponent_agent': agent_B_type,
                        'payoff': row['score_A']
                    })
                    # Add entry for Agent B's score
                    all_payoffs_list.append({
                        'game': game_name,
                        'pairing': descriptive_name,
                        'seed': seed,
                        'meta_round': row['meta_round'],
                        'acting_agent': agent_B_type,
                        'opponent_agent': agent_A_type,
                        'payoff': row['score_B']
                    })
            except Exception as e:
                print(f"Warning: Could not process log file in {run_dir}: {e}")

    if not all_payoffs_list:
        print("\nNo valid log data found for agent-specific payoff aggregation.")
        return None

    combined_df = pd.DataFrame(all_payoffs_list)
    print(f"\nAggregated {len(combined_df)} agent-specific payoff data points.")
    return combined_df

def load_all_llm_judge_data(base_dir: Path, games_to_run: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """
    Finds and aggregates all 'llm_judge_cache.csv' files from individual analyses.
    """
    print(f"\n{'='*20} AGGREGATING LLM-AS-JUDGE DATA {'='*20}")
    cache_files = list(base_dir.rglob('llm_judge_cache.csv'))
    
    if not cache_files:
        print("Error: No 'llm_judge_cache.csv' files found in any subdirectories.")
        return None

    all_llm_data = []
    for cache_file in cache_files:
        # Extract game and pairing info from the path
        try:
            game_name = [p.name for p in cache_file.parents if p.name.startswith('results_')][0].replace('results_', '')
            pairing_name = cache_file.parent.parent.name
            
            # Filter by game if requested
            if games_to_run and not any(game in game_name for game in games_to_run):
                continue
            
            agent_types = _get_agent_types_from_pairing(pairing_name)
            if not agent_types:
                print(f"Warning: Could not determine agent types for '{pairing_name}'. Skipping.")
                continue

            df = pd.read_csv(cache_file)
            df['game'] = game_name
            df['pairing'] = pairing_name
            
            # Map agent 'A'/'B' to actual types like 'CPM'/'DPM'
            df['acting_agent'] = df['agent'].apply(lambda x: agent_types[0] if x == 'A' else agent_types[1])
            df['opponent_agent'] = df['agent'].apply(lambda x: agent_types[1] if x == 'A' else agent_types[0])
            
            all_llm_data.append(df)
        except IndexError:
            print(f"Warning: Could not parse game/pairing for cache file: {cache_file}. Skipping.")
            continue
            
    if not all_llm_data:
        print("No LLM data found for the specified games.")
        return None
        
    print(f"Found and combined LLM data from {len(all_llm_data)} pairing(s).")
    return pd.concat(all_llm_data, ignore_index=True)

def plot_aggregated_strategy_panel(all_llm_data: pd.DataFrame, output_dir: Path):
    """
    Generates a 2x4 panel of grouped bar plots for strategic response proportions.
    """
    print("\nGenerating aggregated strategic response proportion panel...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. --- Data Preparation ---
    one_hot = pd.get_dummies(all_llm_data['classification'])
    one_hot = one_hot.reindex(columns=STRATEGIC_RESPONSE_CATEGORIES, fill_value=0)
    data = pd.concat([all_llm_data, one_hot], axis=1)
    grouping_vars = ['game', 'pairing', 'seed', 'acting_agent', 'opponent_agent']
    proportions_per_run = data.groupby(grouping_vars)[STRATEGIC_RESPONSE_CATEGORIES].mean().reset_index()
    numeric_cols = proportions_per_run.select_dtypes(include=np.number).columns
    final_stats = proportions_per_run.groupby(['game', 'acting_agent', 'opponent_agent'])[numeric_cols].agg(['mean', 'sem'])

    # 2. --- Plotting Setup ---
    response_types = ["Counter_Measure", "Exploitation_Attempt", "Feint", "Direct_Imitation"]
    metric_ylims = {}
    for response_type in response_types:
        try:
            max_val = (final_stats[(response_type, 'mean')] + final_stats[(response_type, 'sem')]).max()
            # Set a minimum limit of 1.0 just in case max is 0
            metric_ylims[response_type] = (0, max(max_val * 100 * 1.1, 1.0)) 
        except KeyError:
            metric_ylims[response_type] = (0, 10) # Default small limit

    agent_groups = ['CPM', 'DPM', 'PM']
    game_map = {
        'CoinGame': {'img_path': 'images/coingame.png', 'row': 0, 'title': 'Coin Game'},
        'IPD': {'img_path': 'images/ipd.png', 'row': 1, 'title': 'IPD'}
    }
    opponent_colors = {'CPM': 'blue', 'DPM': 'red', 'PM': 'black'}

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    # 3. --- Plotting Loop ---
    for game_name, info in game_map.items():
        row = info['row']
        
        axes[row, 0].text(-0.25, 0.5, info['title'], transform=axes[row, 0].transAxes, 
                          fontsize=TITLE_FONTSIZE, va='center', ha='right', rotation=90)
        
        for col, response_type in enumerate(response_types, 0):
            ax = axes[row, col]
            
            bar_width = 0.25
            group_indices = np.arange(len(agent_groups))
            
            for i, opponent_type in enumerate(agent_groups):
                means = []
                sems = []
                for acting_type in agent_groups:
                    try:
                        mean_val = final_stats.loc[(game_name, acting_type, opponent_type)][(response_type, 'mean')]
                        sem_val = final_stats.loc[(game_name, acting_type, opponent_type)][(response_type, 'sem')]
                    except KeyError:
                        mean_val, sem_val = 0, 0
                    means.append(mean_val * 100)
                    sems.append(sem_val * 100)
                
                positions = group_indices + (i - 1) * bar_width
                ax.bar(positions, means, yerr=sems, width=bar_width, 
                       label=f"vs {opponent_type}", color=opponent_colors[opponent_type], capsize=4)

            # Formatting
            ax.set_xticks(group_indices)
            if row == 1:
                ax.set_xticklabels(agent_groups)
                ax.set_xlabel("Actor", fontsize=AXIS_LABEL_FONTSIZE)
            else:
                ax.set_xticklabels([])

            if row == 0:
                ax.set_title(response_type.replace('_', ' '))

            if col == 0:
                ax.set_ylabel("%", fontsize=AXIS_LABEL_FONTSIZE - 2)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            ax.set_ylim(metric_ylims[response_type])

            if row == 0 and col == 0:
                ax.legend(loc='upper right', fontsize=LEGEND_FONTSIZE - 2, title="Opponent")

    # 4. --- Legend ---
    handles, labels = axes[0, 0].get_legend_handles_labels()

    plt.tight_layout(rect=[0.05, 0, 1, 1.0])
    panel_path = output_dir / "strategic_features_panel.pdf"
    plt.savefig(panel_path, bbox_inches='tight')
    print(f"Saved aggregated strategy panel plot to {panel_path}")
    plt.close(fig)

def plot_aggregated_payoff_bars(all_payoff_data: pd.DataFrame, output_dir: Path):
    """
    Generates a 1xN panel of grouped bar plots for average payoff, side-by-side.
    """
    print("\nGenerating aggregated payoff bar panel...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. --- Data Preparation ---
    grouping_vars = ['game', 'pairing', 'seed', 'acting_agent', 'opponent_agent']
    payoffs_per_run = all_payoff_data.groupby(grouping_vars)['payoff'].mean().reset_index()
    final_stats = payoffs_per_run.groupby(['game', 'acting_agent', 'opponent_agent'])['payoff'].agg(['mean', 'sem'])

    # 2. --- Plotting Setup ---
    games = sorted(list(all_payoff_data['game'].unique()))
    agent_groups = ['CPM', 'DPM', 'PM']
    
    game_map = {
        'CoinGame': {'img_path': 'images/coingame.png', 'title': 'Coin Game'},
        'IPD': {'img_path': 'images/ipd.png', 'title': 'IPD'}
    }
    game_map = {g: info for g, info in game_map.items() if g in games}
    num_games = len(game_map)
    
    if num_games == 0:
        print("No games found to plot for payoff bars.")
        return
    
    opponent_colors = {'CPM': 'blue', 'DPM': 'red', 'PM': 'black'}

    fig, axes = plt.subplots(1, num_games * 2, figsize=(7 * num_games, 5), 
                             gridspec_kw={'width_ratios': [3, 4] * num_games}, squeeze=False)
    axes = axes.flatten() # Flatten to 1D array

    # 3. --- Plotting Loop ---
    for i, (game_name, info) in enumerate(game_map.items()):
        img_col = i * 2
        plot_col = i * 2 + 1
        
        # Plot game image
        ax_img = axes[img_col]
        try:
            img = plt.imread(info['img_path'])
            ax_img.imshow(img)
            ax_img.set_title(info['title'], fontsize=TITLE_FONTSIZE, pad=15)
        except FileNotFoundError:
            ax_img.text(0.5, 0.5, f'Image not found', ha='center', va='center')
            ax_img.set_title(info['title'], fontsize=TITLE_FONTSIZE, pad=15)
        ax_img.axis('off')

        # Plot bar chart
        ax = axes[plot_col]
        
        bar_width = 0.25
        group_indices = np.arange(len(agent_groups))
        
        for j, opponent_type in enumerate(agent_groups):
            means = []
            sems = []
            for acting_type in agent_groups:
                try:
                    mean_val = final_stats.loc[(game_name, acting_type, opponent_type)]['mean']
                    sem_val = final_stats.loc[(game_name, acting_type, opponent_type)]['sem']
                except KeyError:
                    mean_val, sem_val = 0, 0
                means.append(mean_val)
                sems.append(sem_val)
            
            positions = group_indices + (j - 1) * bar_width
            ax.bar(positions, means, yerr=sems, width=bar_width, 
                   label=f"vs {opponent_type}", color=opponent_colors.get(opponent_type, 'gray'), capsize=4)

        # Formatting
        ax.set_xticks(group_indices)
        ax.set_xticklabels(agent_groups)
        ax.set_xlabel("Actor", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Average Payoff", fontsize=AXIS_LABEL_FONTSIZE)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if i == 0: # Add legend to the first plot
            ax.legend(loc='upper right', fontsize=LEGEND_FONTSIZE - 2, title="Opponent")

    plt.tight_layout()
    panel_path = output_dir / "aggregated_payoffs_bar_panel.pdf"
    plt.savefig(panel_path, bbox_inches='tight')
    print(f"Saved aggregated payoff bar panel plot to {panel_path}")
    plt.close(fig)

def generate_strategy_proportion_table(all_llm_data: pd.DataFrame, output_dir: Path):
    """
    Calculates and prints an aggregate table of mean strategic mechanism
    proportions, broken down by game and meta-round.
    """
    print(f"\n{'='*20} AGGREGATE STRATEGY PROPORTION TABLE {'='*20}")
    
    if all_llm_data is None or all_llm_data.empty:
        print("No LLM data found to generate table.")
        return

    # 1. One-hot encode the classifications
    one_hot = pd.get_dummies(all_llm_data['classification'])
    # Reindex to ensure all possible category columns exist
    one_hot = one_hot.reindex(columns=STRATEGIC_RESPONSE_CATEGORIES, fill_value=0)
    
    # 2. Combine with grouping keys
    data = pd.concat([all_llm_data[['game', 'pairing', 'seed', 'meta_round']], one_hot], axis=1)
    
    # 3. Calculate proportions *within* each run/round
    # This averages the one-hot columns (e.g., 2 agents, 1 is 'Counter') -> mean = 0.5
    proportions_per_run_round = data.groupby(['game', 'pairing', 'seed', 'meta_round']).mean(numeric_only=True)
    
    # 4. Calculate the average of these proportions across all runs
    # This gives the final mean proportion for a given game and round
    avg_proportions_by_round = proportions_per_run_round.groupby(['game', 'meta_round']).mean()
    
    # 5. Format the table as requested
    
    # Stack the classification columns (e.g., 'Counter_Measure') into a new index level
    stacked_data = avg_proportions_by_round.stack()
    stacked_data.name = 'avg_proportion'
    
    # Unstack the 'meta_round' to turn it into columns
    final_table = stacked_data.unstack('meta_round')
    
    # Convert to percentages
    final_table_pct = final_table * 100
    
    # 6. Print the formatted table
    print("Average Proportion of Strategic Mechanism (%) by Game and Meta-Round")
    print("-" * 80)
    
    for game in final_table_pct.index.get_level_values('game').unique():
        print(f"\nGame: {game}\n")
        game_table = final_table_pct.loc[game]
        # Fill NaN with 0.0 for cleaner output
        game_table = game_table.fillna(0.0)
        
        # Format the column headers to be integers (e.g., '1' instead of '1.0')
        game_table.columns = game_table.columns.astype(int)
        
        print(game_table.to_string(float_format="%.2f%%"))
        print("-" * 80)

    # Optionally, save to CSV
    csv_path = output_dir / "aggregate_strategy_proportions.csv"
    final_table_pct.to_csv(csv_path)
    print(f"Table also saved to: {csv_path}")

def plot_logged_payoffs_over_time(all_log_data: List[pd.DataFrame], output_dir: Path):
    """Plots the average final score from logs over meta-rounds."""
    if not all_log_data:
        print("No log data provided for payoff plotting.")
        return

    combined_df = pd.concat(all_log_data, ignore_index=True)
    grouped = combined_df.groupby('meta_round')
    
    mean_df = grouped[['score_A', 'score_B']].mean()
    sem_df = grouped[['score_A', 'score_B']].sem()

    plt.figure(figsize=(10, 6))
    
    agent_A_label = combined_df['agent_A'].iloc[0]
    agent_B_label = combined_df['agent_B'].iloc[0]

    for agent_id, label in [('A', agent_A_label), ('B', agent_B_label)]:
        mean_vals = mean_df[f'score_{agent_id}']
        sem_vals = sem_df[f'score_{agent_id}']
        
        plt.plot(mean_df.index, mean_vals, marker='o', linestyle='-', label=f"{label}")
        plt.fill_between(mean_df.index, 
                         mean_vals - sem_vals, 
                         mean_vals + sem_vals, 
                         alpha=0.2)

    plt.title("Average Score from Logs vs. Meta-Round", fontsize=TITLE_FONTSIZE)
    plt.xlabel("Meta-Round", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Average Score", fontsize=AXIS_LABEL_FONTSIZE)
    plt.xticks(np.arange(min(mean_df.index), max(mean_df.index) + 1, 1))
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_path = output_dir / "dyadic_logged_scores.pdf"
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    plt.close()

def generate_individual_payoff_plot(mean_df: pd.DataFrame, sem_df: pd.DataFrame, per_experiment_data: pd.DataFrame, output_dir: Path):
    """
    Generates and saves a single plot for aggregated payoffs for a specific game.
    """
    print(f"Generating individual payoff plot in {output_dir}...")
    game_name = output_dir.name.replace('_aggregated_metrics', '')

    # Color mapping
    unique_experiments = per_experiment_data.index.get_level_values('experiment').unique()
    label_map = {name: extract_pairing_name(name) for name in unique_experiments}
    unique_labels = sorted(list(set(label_map.values())))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    label_color_map = dict(zip(unique_labels, colors))
    experiment_color_map = {exp_name: label_color_map[clean_label] for exp_name, clean_label in label_map.items()}

    plt.figure(figsize=(10, 6))
    
    y = mean_df['payoff']
    X = sm.add_constant(mean_df.index)
    model = sm.OLS(y, X).fit()
    p_value = model.pvalues[1]

    for experiment_name, experiment_df in per_experiment_data.groupby(level='experiment'):
        exp_metric_data = experiment_df['payoff']
        plt.plot(exp_metric_data.index.get_level_values('meta_round'), exp_metric_data.values,
                 marker='.', linestyle='-', color=experiment_color_map.get(experiment_name, 'gray'), alpha=0.3, zorder=1)

    mean_vals = mean_df['payoff']
    sem_vals = sem_df['payoff']
    plt.plot(mean_df.index, mean_vals, marker='o', linestyle='-', label='Game-wide Average', color='black', zorder=2)
    plt.fill_between(mean_df.index, mean_vals - sem_vals, mean_vals + sem_vals, color='black', alpha=0.2, zorder=2)
    
    plt.text(0.95, 0.95, f'p = {p_value:.3f}', transform=plt.gca().transAxes,
             fontsize=14, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    plt.title(f"Aggregated Average Score Over Time for {game_name}", fontsize=TITLE_FONTSIZE)
    plt.xlabel("Meta-Round", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Average Score", fontsize=AXIS_LABEL_FONTSIZE)
    plt.xticks(np.arange(min(mean_df.index), max(mean_df.index) + 1, 1))
    
    legend_handles = [plt.Line2D([0], [0], color=color, lw=2) for label, color in sorted(label_color_map.items())]
    plt.legend(handles=legend_handles, labels=sorted(label_color_map.keys()), title="Pairings", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_path = output_dir / "aggregated_payoffs_over_time.pdf"
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"  - Saved plot: {plot_path.name}")
    plt.close()


def plot_aggregated_payoff_plot(mean_df: pd.DataFrame, sem_df: pd.DataFrame, per_experiment_data: pd.DataFrame, output_dir: Path):
    """
    Generates and saves a single plot for aggregated payoffs.
    """
    print(f"Generating aggregated payoff plot in {output_dir}...")
    
    # Color mapping
    unique_experiments = per_experiment_data.index.get_level_values('experiment').unique()
    label_map = {name: extract_pairing_name(name) for name in unique_experiments}
    unique_labels = sorted(list(set(label_map.values())))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    label_color_map = dict(zip(unique_labels, colors))
    experiment_color_map = {exp_name: label_color_map[clean_label] for exp_name, clean_label in label_map.items()}

    plt.figure(figsize=(10, 6))
    
    y = mean_df['payoff']
    X = sm.add_constant(mean_df.index)
    model = sm.OLS(y, X).fit()
    p_value = model.pvalues[1]

    for experiment_name, experiment_df in per_experiment_data.groupby(level='experiment'):
        exp_metric_data = experiment_df['payoff']
        plt.plot(exp_metric_data.index.get_level_values('meta_round'), exp_metric_data.values,
                 marker='.', linestyle='-', color=experiment_color_map.get(experiment_name, 'gray'), alpha=0.3, zorder=1)

    mean_vals = mean_df['payoff']
    sem_vals = sem_df['payoff']
    plt.plot(mean_df.index, mean_vals, marker='o', linestyle='-', label='Game-wide Average', color='black', zorder=2)
    plt.fill_between(mean_df.index, mean_vals - sem_vals, mean_vals + sem_vals, color='black', alpha=0.2, zorder=2)
    
    plt.text(0.95, 0.95, f'p = {p_value:.3f}', transform=plt.gca().transAxes,
             fontsize=14, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    plt.title("Aggregated Average Score Over Time", fontsize=TITLE_FONTSIZE)
    plt.xlabel("Meta-Round", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Average Score", fontsize=AXIS_LABEL_FONTSIZE)
    plt.xticks(np.arange(min(mean_df.index), max(mean_df.index) + 1, 1))
    
    legend_handles = [plt.Line2D([0], [0], color=color, lw=2) for label, color in sorted(label_color_map.items())]
    plt.legend(handles=legend_handles, labels=sorted(label_color_map.keys()), title="Pairings", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_path = output_dir / "aggregated_payoffs_over_time.pdf"
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"  - Saved plot: {plot_path.name}")
    plt.close()


def run_llm_as_judge(all_runs_data: List[pd.DataFrame], output_dir: Path, client: openai.OpenAI):
    """Classifies strategic responses using an LLM and generates visualizations."""
    if not client:
        print("Skipping LLM-as-Judge analysis: OpenAI client not available.")
        return

    cache_path = output_dir / "llm_judge_cache.csv"
    if cache_path.exists():
        print(f"Loading LLM classifications from cache: {cache_path}")
        results_df = pd.read_csv(cache_path)
    else:
        results = []
        all_runs_data_reset = [df.reset_index(drop=True) for df in all_runs_data]
        for i, run_df in enumerate(all_runs_data_reset):
            print(f"Running LLM-as-Judge on seed {i+1}/{len(all_runs_data_reset)}...")
            for meta_round in range(1, run_df['meta_round'].max() + 1):
                current_row_series = run_df[run_df['meta_round'] == meta_round]
                if current_row_series.empty: continue
                current_row = current_row_series.iloc[0]

                prev_row_series = run_df[run_df['meta_round'] == meta_round - 1]
                prev_row = prev_row_series.iloc[0] if not prev_row_series.empty else None

                for agent_curr, agent_prev in [('A', 'B'), ('B', 'A')]:
                    text_t = load_file_content(Path(current_row[f'text_{agent_curr}_path']))
                    code_t = load_file_content(Path(current_row[f'code_{agent_curr}_path']))
                    text_t_minus_1 = load_file_content(Path(prev_row[f'text_{agent_prev}_path'])) if prev_row is not None else None
                    code_t_minus_1 = load_file_content(Path(prev_row[f'code_{agent_prev}_path'])) if prev_row is not None else None
                    
                    classification, rationale = get_llm_classification(
                        client=client,
                        agent_strategy_t=text_t, agent_code_t=code_t,
                        opponent_strategy_t_minus_1=text_t_minus_1, opponent_code_t_minus_1=code_t_minus_1,
                        player_id=agent_curr, current_meta_round=meta_round
                    )
                    
                    results.append({'seed': i, 'meta_round': meta_round, 'agent': agent_curr, 'classification': classification, 'rationale': rationale})
                    
        results_df = pd.DataFrame(results)
        results_df.to_csv(cache_path, index=False)
        print(f"Saved LLM classifications to cache: {cache_path}")

    if results_df.empty:
        print("No LLM judge results to plot.")
        return

    # Visualization 1: Stacked Bar Charts
    pivot = results_df.groupby(['meta_round', 'agent', 'classification']).size().unstack(fill_value=0)
    proportions = pivot.div(pivot.sum(axis=1), axis=0).reset_index()

    for agent_id in ['A', 'B']:
        agent_data = proportions[proportions['agent'] == agent_id]
        if agent_data.empty: continue
        
        plt.figure(figsize=(12, 7))
        bottom = np.zeros(agent_data['meta_round'].nunique())
        agent_data_plot = agent_data.set_index('meta_round')
        all_rounds = np.arange(1, results_df['meta_round'].max() + 1)
        
        for category in STRATEGIC_RESPONSE_CATEGORIES:
            if category not in agent_data_plot.columns: agent_data_plot[category] = 0
        for category in STRATEGIC_RESPONSE_CATEGORIES:
            values = agent_data_plot[category].reindex(all_rounds, fill_value=0)
            plt.bar(values.index, values.values, label=category, bottom=bottom, color=STRATEGY_PLOT_COLORS.get(category))
            bottom += values.values

        agent_label = all_runs_data[0][f'agent_{agent_id}'].iloc[0]
        plt.title(f"Strategic Response Proportions for Agent {agent_id} ({agent_label})", fontsize=TITLE_FONTSIZE)
        plt.xlabel("Meta-Round", fontsize=AXIS_LABEL_FONTSIZE)
        plt.ylabel("Proportion", fontsize=AXIS_LABEL_FONTSIZE)
        plt.xticks(all_rounds)
        plt.legend(title="Strategy Type", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plot_path = output_dir / f"dyadic_llm_judge_stacked_bar_agent_{agent_id}.pdf"
        plt.savefig(plot_path)
        plt.close()

    # Visualization 2: All Line Plots and Panels
    _plot_llm_judge_line_plots(results_df, all_runs_data, output_dir)
    _plot_llm_judge_full_panel(results_df, all_runs_data, output_dir)
    _plot_llm_judge_custom_panel(results_df, all_runs_data, output_dir)

def load_and_compile_programs(run_dir: Path, game: harness.Game) -> bool:
    """
    Finds all .py files in a run's program_strategies folder, compiles them,
    and populates the global PROGRAMS dictionary from the harness.
    """
    harness.PROGRAMS.clear()
    prog_dir = run_dir / "program_strategies"
    if not prog_dir.exists():
        print(f"Warning: Program strategies directory not found in {run_dir}")
        return False

    py_files = list(prog_dir.glob("*.py"))
    exec_ns = game.get_execution_namespace()
    
    for py_file in py_files:
        try:
            prog_name = py_file.stem
            code = py_file.read_text()
            ast.parse(code) # Check for syntax errors
            
            func_match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code)
            if not func_match: continue
            func_name = func_match.group(1)

            local_ns = exec_ns.copy()
            exec(code, local_ns)
            
            harness.PROGRAMS[prog_name] = {
                'function': local_ns[func_name],
                'code': code,
                'agent_label': prog_name.split('_MR')[0]
            }
        except Exception as e:
            print(f"Warning: Could not compile program {py_file.name}. Error: {e}")
            
    if not harness.PROGRAMS:
        print(f"Warning: No programs were successfully compiled for run {run_dir.name}.")
        return False
        
    return True

def analyze_match_dynamics(pairing_dir: Path) -> Optional[pd.DataFrame]:
    """
    Runs simulations for all seeds in a pairing to get payoff & action data.
    """
    print("\n--- Running Match Dynamics Analysis (Payoffs & Cooperation) ---")
    run_dirs = find_experiment_runs(pairing_dir)
    if not run_dirs:
        print("No run directories found for match dynamics analysis.")
        return None

    # Determine the game type from the folder name
    dir_name = pairing_dir.parent.name
    if 'IPD' in dir_name:
        game = harness.IPDGame(rounds=100) # Assuming 100 rounds for accurate stats
    elif 'CoinGame' in dir_name:
        game = harness.CoinGame(max_steps=50, board_size=3)
    else:
        print(f"Warning: Could not determine game type for {pairing_dir}. Skipping dynamics analysis.")
        return None

    all_run_results = []
    for run_dir in run_dirs:
        print(f"  Processing seed: {run_dir.name}")
        log_file = next(run_dir.glob("log_seed_*.csv"), None)
        if not log_file:
            print(f"    Warning: No log file found in {run_dir.name}. Skipping.")
            continue
        
        # Compile all programs for this seed run
        if not load_and_compile_programs(run_dir, game):
            continue
            
        log_df = pd.read_csv(log_file)
        
        for _, row in log_df.iterrows():
            meta_round = row['meta_round']
            prog_A, prog_B = row['prog_A'], row['prog_B']
            
            if prog_A not in harness.PROGRAMS or prog_B not in harness.PROGRAMS:
                print(f"    Skipping MR {meta_round}: one or both programs missing from compilation.")
                continue

            match_result = game.run_match(prog_A, prog_B)
            
            # Normalize payoffs by match duration
            duration = getattr(game, 'rounds', getattr(game, 'max_steps', 1))
            avg_payoff_A = match_result.get('score_A', 0) / duration
            avg_payoff_B = match_result.get('score_B', 0) / duration
            
            result_row = {
                'seed': int(run_dir.name.split('_')[-1]),
                'meta_round': meta_round,
                'agent_A_label': row['agent_A'],
                'agent_B_label': row['agent_B'],
                'avg_payoff_A': avg_payoff_A,
                'avg_payoff_B': avg_payoff_B,
                'coop_rate_A': np.nan,
                'defect_rate_A': np.nan,
                'coop_rate_B': np.nan,
                'defect_rate_B': np.nan,
            }

            if isinstance(game, harness.IPDGame):
                hist_A = match_result.get('history_A', [])
                hist_B = match_result.get('history_B', [])
                if hist_A:
                    counts_A = Counter(hist_A)
                    result_row['coop_rate_A'] = counts_A.get('C', 0) / len(hist_A)
                    result_row['defect_rate_A'] = counts_A.get('D', 0) / len(hist_A)
                if hist_B:
                    counts_B = Counter(hist_B)
                    result_row['coop_rate_B'] = counts_B.get('C', 0) / len(hist_B)
                    result_row['defect_rate_B'] = counts_B.get('D', 0) / len(hist_B)
            
            all_run_results.append(result_row)
    
    if not all_run_results:
        return None

    # Aggregate results across seeds
    results_df = pd.DataFrame(all_run_results)
    grouped = results_df.groupby('meta_round')
    
    # --- START CORRECTION ---
    # Specify numeric_only=True to ignore string columns like agent labels
    mean_df = grouped.mean(numeric_only=True)
    sem_df = grouped.sem(numeric_only=True)
    # --- END CORRECTION ---
    
    # Combine into a single DataFrame for easier plotting
    final_df = mean_df.merge(sem_df, on='meta_round', suffixes=('_mean', '_sem'))
    # Preserve agent labels for plotting titles/legends
    final_df['agent_A_label'] = results_df.groupby('meta_round')['agent_A_label'].first()
    final_df['agent_B_label'] = results_df.groupby('meta_round')['agent_B_label'].first()
    
    return final_df.drop(columns=['seed_mean', 'seed_sem'])


def plot_payoffs_over_time(dynamics_df: pd.DataFrame, output_dir: Path):
    """Plots the average payoff per turn over meta-rounds."""
    plt.figure(figsize=(10, 6))
    
    agent_A_label = dynamics_df['agent_A_label'].iloc[0]
    agent_B_label = dynamics_df['agent_B_label'].iloc[0]

    for agent_id, label in [('A', agent_A_label), ('B', agent_B_label)]:
        mean_col = f'avg_payoff_{agent_id}_mean'
        sem_col = f'avg_payoff_{agent_id}_sem'
        
        plt.plot(dynamics_df.index, dynamics_df[mean_col], marker='o', linestyle='-', label=f"{label}")
        plt.fill_between(dynamics_df.index, 
                         dynamics_df[mean_col] - dynamics_df[sem_col], 
                         dynamics_df[mean_col] + dynamics_df[sem_col], 
                         alpha=0.2)

    plt.title("Average Payoff per Turn vs. Meta-Round", fontsize=TITLE_FONTSIZE)
    plt.xlabel("Meta-Round", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Average Payoff per Turn", fontsize=AXIS_LABEL_FONTSIZE)
    plt.xticks(np.arange(min(dynamics_df.index), max(dynamics_df.index) + 1, 1))
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_path = output_dir / "dyadic_avg_payoffs.pdf"
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    plt.close()


def plot_ipd_behavior_over_time(dynamics_df: pd.DataFrame, output_dir: Path):
    """Plots cooperation and defection rates over meta-rounds for IPD."""
    if 'coop_rate_A_mean' not in dynamics_df.columns or dynamics_df['coop_rate_A_mean'].isnull().all():
        return # Skip if no cooperation data is available

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle("IPD Action Rates vs. Meta-Round", fontsize=TITLE_FONTSIZE)
    
    agent_A_label = dynamics_df['agent_A_label'].iloc[0]
    agent_B_label = dynamics_df['agent_B_label'].iloc[0]
    
    # Plot Cooperation Rate
    ax1.set_title("Cooperation Rate")
    for agent_id, label in [('A', agent_A_label), ('B', agent_B_label)]:
        mean_col = f'coop_rate_{agent_id}_mean'
        sem_col = f'coop_rate_{agent_id}_sem'
        ax1.plot(dynamics_df.index, dynamics_df[mean_col], marker='o', linestyle='-', label=f"{label}")
        ax1.fill_between(dynamics_df.index, 
                         dynamics_df[mean_col] - dynamics_df[sem_col], 
                         dynamics_df[mean_col] + dynamics_df[sem_col], 
                         alpha=0.2)

    # Plot Defection Rate
    ax2.set_title("Defection Rate")
    for agent_id, label in [('A', agent_A_label), ('B', agent_B_label)]:
        mean_col = f'defect_rate_{agent_id}_mean'
        sem_col = f'defect_rate_{agent_id}_sem'
        ax2.plot(dynamics_df.index, dynamics_df[mean_col], marker='o', linestyle='-', label=f"{label}")
        ax2.fill_between(dynamics_df.index, 
                         dynamics_df[mean_col] - dynamics_df[sem_col], 
                         dynamics_df[mean_col] + dynamics_df[sem_col], 
                         alpha=0.2)

    for ax in [ax1, ax2]:
        ax.set_xlabel("Meta-Round", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=LEGEND_FONTSIZE)
    ax1.set_ylabel("Proportion of Actions", fontsize=AXIS_LABEL_FONTSIZE)
    
    plot_path = output_dir / "dyadic_ipd_behavior.pdf"
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    plt.close()

def get_llm_classification(
    client: openai.OpenAI,
    agent_strategy_t: str,
    agent_code_t: str,
    opponent_strategy_t_minus_1: str,
    opponent_code_t_minus_1: str,
    player_id: str,
    current_meta_round: int
) -> Tuple[str, str]:
    """
    Classifies an agent's strategic response using a detailed prompt.
    """
    if not agent_strategy_t and not agent_code_t:
        return "Data_Missing", "Agent's strategy and code for current round are missing."

    # Handle first round where opponent has no history
    if current_meta_round == 1:
        opponent_strategy_t_minus_1 = opponent_strategy_t_minus_1 or "No previous strategy (first meta-round)."
        opponent_code_t_minus_1 = opponent_code_t_minus_1 or "# No previous code (first meta-round)."

    opponent_id = 'B' if player_id == 'A' else 'A'
    prompt = f"""
You are an expert AI agent analyst specializing in multi-round strategic games.
Your task is to classify Agent {player_id}'s strategic approach in the current meta-round (t) based on its textual strategy and generated code, in relation to the opponent's (Agent {opponent_id}) strategy and code from the *previous* meta-round (t-1).

**Input Data:**
1.  **Agent {player_id}'s Textual Strategy (Meta-Round t):**
    ```text
    {agent_strategy_t or "Not Available"}
    ```

2.  **Agent {player_id}'s Generated Code (Meta-Round t):**
    ```python
    {agent_code_t or "# Not Available"}
    ```

3.  **Opponent's (Agent {opponent_id}) Textual Strategy (Meta-Round t-1):**
    ```text
    {opponent_strategy_t_minus_1 or "Not Available"}
    ```

4.  **Opponent's (Agent {opponent_id}) Generated Code (Meta-Round t-1):**
    ```python
    {opponent_code_t_minus_1 or "# Not Available"}
    ```

**Classification Categories & Definitions:**
* **Independent Development:** The agent's code/strategy shows no clear, direct reactive link to the opponent's t-1 materials.
* **Direct Imitation:** The agent's code/strategy significantly incorporates or copies core logic from the opponent's t-1 materials.
* **Counter Measure:** The agent's code/strategy is primarily designed to neutralize or defend against the opponent's t-1 strategy.
* **Exploitation Attempt:** The agent's code/strategy attempts to take advantage of a perceived weakness in the opponent's t-1 strategy.
* **Feint:** The agent's code/strategy seems primarily designed to mislead the opponent, perhaps with mismatched comments or logic.

**Task:**
Analyze the provided data. Respond with a JSON object containing two keys: "classification" (one of the categories above, using underscores like "Independent_Development") and "rationale" (your explanation for your classification). Do not include any other text before or after the JSON object.
"""
    try:
        completion = client.chat.completions.create(
            model=LLM_JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert analyst of AI agent strategies in games."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=250,
            response_format={"type": "json_object"}
        )
        response_content = completion.choices[0].message.content
        response_json = json.loads(response_content)
        classification = response_json.get("classification")
        rationale = response_json.get("rationale")

        if classification not in STRATEGIC_RESPONSE_CATEGORIES:
            print(f"Warning: LLM returned invalid classification '{classification}'. Defaulting to Independent_Development.")
            return "Independent_Development", f"Invalid classification '{classification}'. Rationale: {rationale}"
        return classification, rationale
    except json.JSONDecodeError:
        print(f"Error: LLM response was not valid JSON: {response_content}")
        return "Error", "LLM response was not valid JSON."
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "Error", str(e)

# --- Evolutionary Analysis ---

# --- EGT Simplex Plotting Helpers (from simplex.py) ---
simplex_r0 = np.array([0, 0])
simplex_r1 = np.array([1, 0])
simplex_r2 = np.array([0.5, np.sqrt(3)/2.])
simplex_corners = np.array([simplex_r0, simplex_r1, simplex_r2])
simplex_triangle = tri.Triangulation(simplex_corners[:, 0], simplex_corners[:, 1])

# --- Simplex Figure Size Constants ---
SIMPLEX_FIGSIZE = (14, 13)  # Standardized figure size for all simplex plots
SIMPLEX_PANEL_FIGSIZE = (28, 13)  # Figure size for side-by-side simplex panel

# --- Simplex Dynamics Plotting Class (from simplex.py) ---
class SimplexDynamicsPlotter:
    r0, r1, r2 = np.array([0,0]), np.array([1,0]), np.array([0.5, np.sqrt(3)/2.])
    corners = np.array([r0, r1, r2])
    triangle = tri.Triangulation(corners[:,0], corners[:,1])
    try:
        refiner = tri.UniformTriRefiner(triangle)
        trimesh = refiner.refine_triangulation(subdiv=5) # Mesh density
    except Exception as e:
        print(f"Warning: Could not initialize trimesh: {e}")
        refiner = None
        trimesh = None

    def __init__(self, replicator_func: Callable, strategy_labels: List[str], corner_label_fontsize: int = 28):
        self.f = replicator_func
        self.strategy_labels = strategy_labels
        self.corner_label_fontsize = corner_label_fontsize
        if self.trimesh is None: # Fallback initialization
            print("Info: Initializing trimesh in __init__.")
            SimplexDynamicsPlotter.refiner = tri.UniformTriRefiner(self.triangle)
            SimplexDynamicsPlotter.trimesh = SimplexDynamicsPlotter.refiner.refine_triangulation(subdiv=5)
        self.calculate_stationary_points()
        self.calc_direction_and_strength()

    def xy2ba(self, x,y):
        detT = (self.corners[1,1]-self.corners[2,1])*(self.corners[0,0]-self.corners[2,0]) + \
               (self.corners[2,0]-self.corners[1,0])*(self.corners[0,1]-self.corners[2,1])
        if abs(detT)<1e-12: return np.array([np.nan]*3)
        l1 = ((self.corners[1,1]-self.corners[2,1])*(x-self.corners[2,0]) + \
              (self.corners[2,0]-self.corners[1,0])*(y-self.corners[2,1]))/detT
        l2 = ((self.corners[2,1]-self.corners[0,1])*(x-self.corners[2,0]) + \
              (self.corners[0,0]-self.corners[2,0])*(y-self.corners[2,1]))/detT
        return np.array([l1,l2,1-l1-l2])

    def ba2xy(self, ba):
        ba=np.array(ba)
        return self.corners.T.dot(ba.T).T if ba.ndim > 1 else self.corners.T.dot(ba)

    def calculate_stationary_points(self, tol=1e-8, margin=0.005):
        fp_bary = []
        if self.trimesh is None:
            print("Warning: trimesh not available. Skipping fixed point calculation.")
            self.fixpoints = np.array([])
            return
        for x_coord,y_coord in zip(self.trimesh.x, self.trimesh.y):
            start_ba = self.xy2ba(x_coord,y_coord)
            if np.any(start_ba < margin) or np.any(np.isnan(start_ba)): continue
            try:
                sol = scipy.optimize.root(lambda vec: self.f(vec,0), start_ba, method="hybr", tol=tol)
                if sol.success and math.isclose(np.sum(sol.x),1,abs_tol=1e-3) and \
                   np.all((sol.x > -1e-12)&(sol.x < 1+1e-12)):
                    if not any(np.allclose(sol.x, fp, atol=1e-5) for fp in fp_bary):
                        fp_bary.append(sol.x.tolist())
            except Exception: continue
        self.fixpoints = self.ba2xy(np.array(fp_bary)) if fp_bary else np.array([])
        print(f"Found {len(fp_bary)} fixed points (barycentric): {fp_bary}")

    def calc_direction_and_strength(self):
        if self.trimesh is None:
            print("Warning: trimesh not available. Skipping flow field calculation.")
            self.pvals = np.array([]); self.dir_norm_xy = np.array([])
            return
        bary = np.array([self.xy2ba(x,y) for x,y in zip(self.trimesh.x, self.trimesh.y)])
        dir_ba = np.array([self.f(ba,0) if not np.any(np.isnan(ba)) else [0,0,0] for ba in bary])
        self.pvals = np.linalg.norm(dir_ba, axis=1)
        next_bary = np.clip(bary + dir_ba * 0.1, 0, 1)
        next_bary_sum = np.sum(next_bary, axis=1, keepdims=True)
        next_bary = np.divide(next_bary, next_bary_sum, out=np.full_like(next_bary, 1/3.), where=next_bary_sum!=0)
        curr_xy = self.ba2xy(bary); next_xy = self.ba2xy(next_bary)
        self.dir_xy = next_xy - curr_xy; norms = np.linalg.norm(self.dir_xy, axis=1)
        self.dir_norm_xy = np.divide(self.dir_xy, norms[:,np.newaxis], out=np.zeros_like(self.dir_xy), where=norms[:,np.newaxis]!=0)

    def plot_dynamics_simplex(self, ax, cmap='viridis', colorbar_label_fontsize: int = 22, show_colorbar_label: bool = True, **kwargs):
        ax.set_facecolor('white'); ax.triplot(self.triangle, lw=0.8, c="darkgrey", zorder=1)
        if self.trimesh is None:
            print("Warning: trimesh not available. Plot will be minimal.")
        else:
            if hasattr(self,'pvals') and self.pvals.size > 0:
                contour = ax.tricontourf(self.trimesh, self.pvals, alpha=0.6, cmap=cmap, levels=14, zorder=2, **kwargs)
                cb = plt.colorbar(contour, ax=ax, shrink=0.81)
                if show_colorbar_label:
                    cb.set_label("Flow Strength", fontsize=colorbar_label_fontsize)
                cb.ax.tick_params(labelsize=colorbar_label_fontsize - 4)
            if hasattr(self,'dir_norm_xy') and self.dir_norm_xy.size > 0 :
                ax.quiver(self.trimesh.x, self.trimesh.y, self.dir_norm_xy[:,0], self.dir_norm_xy[:,1],
                          angles='xy', pivot='mid', scale=20, width=0.004, headwidth=3.5, color='black', zorder=3)
        if hasattr(self,'fixpoints') and self.fixpoints.size > 0:
            ax.scatter(self.fixpoints[:,0], self.fixpoints[:,1], c="red", s=150, marker='o',
                       edgecolors='black', lw=1.2, zorder=5, label="Fixed Points")
        mgn = 0.05
        ax.text(self.r0[0], self.r0[1]-mgn, self.strategy_labels[0], ha='center',va='top',fontsize=self.corner_label_fontsize)
        ax.text(self.r1[0], self.r1[1]-mgn, self.strategy_labels[1], ha='center',va='top',fontsize=self.corner_label_fontsize)
        ax.text(self.r2[0], self.r2[1]+mgn*0.5, self.strategy_labels[2], ha='center',va='bottom',fontsize=self.corner_label_fontsize)

        ax.axis('equal'); ax.axis('off')
        ax.set_ylim(ymin=-0.1, ymax=self.r2[1]+0.1); ax.set_xlim(xmin=-0.1, xmax=1.1)

def plot_simplex_from_tournament_data(
    payoff_matrix_path: Path,
    population_history_path: Path,
    output_dir: Path,
    tournament_run_name: str,
    plot_title: Optional[str] = None,
    show_colorbar_label: bool = True,
    corner_label_fontsize: int = 32,
    colorbar_label_fontsize: int = 28,
    main_title_fontsize: int = 34,
    legend_fontsize: int = 26,
    trajectory_marker_size: int = 8,
    trajectory_line_width: float = 3.0,
    start_end_marker_size: int = 14,
    show_trajectory: bool = False,
    figsize: tuple = SIMPLEX_FIGSIZE
    ):
    """
    Generates and saves a simplex plot using data from tournament outputs.
    (Adapted from simplex.py)
    """
    print(f"\n--- Generating Simplex Plot for Tournament: {tournament_run_name} ---")
    print(f"  Reading payoff matrix: {payoff_matrix_path}")
    if show_trajectory:
        print(f"  Reading population history for trajectory: {population_history_path}")

    try:
        payoff_df = pd.read_csv(payoff_matrix_path, index_col=0) # Set index_col=0
        strategy_labels = list(payoff_df.columns)
        payoff_matrix = payoff_df.to_numpy()
        if len(strategy_labels) != 3 or payoff_matrix.shape != (3,3):
            print(f"Error: Simplex plot requires 3 strategies. Found {len(strategy_labels)}."); return
    except FileNotFoundError:
        print(f"Error: Payoff matrix file not found: {payoff_matrix_path}"); return
    except Exception as e:
        print(f"Error loading payoff matrix {payoff_matrix_path}: {e}"); return

    population_history_data = None
    if show_trajectory:
        try:
            pop_history_df = pd.read_csv(population_history_path)
            if not all(label in pop_history_df.columns for label in strategy_labels):
                print(f"Warning: Population history CSV missing columns for labels: {strategy_labels}. Found: {list(pop_history_df.columns)}");
                show_trajectory = False # Turn off trajectory
            else:
                population_history_data = pop_history_df[strategy_labels].to_numpy()
        except FileNotFoundError:
            print(f"Warning: Population history file not found: {population_history_path}. Plotting without trajectory.")
            show_trajectory = False
        except Exception as e:
            print(f"Error loading population history {population_history_path}: {e}")
            show_trajectory = False

    def replicator_dyn(x_props, t, A_matrix):
        x_props = np.clip(np.array(x_props), 0, 1)
        x_sum = np.sum(x_props)
        x_props = x_props / x_sum if x_sum > 1e-9 else np.full_like(x_props, 1/len(x_props))
        expected_payoffs = A_matrix.dot(x_props)
        average_population_payoff = x_props.dot(expected_payoffs)
        return x_props * (expected_payoffs - average_population_payoff)

    fig, ax = plt.subplots(figsize=figsize)
    if plot_title:
        ax.text(0.08, 0.5, plot_title, transform=ax.transAxes, 
                fontsize=main_title_fontsize, va='center', ha='right', 
                rotation=90, fontfamily='monospace', weight='bold')
    plotter = None
    try:
        plotter = SimplexDynamicsPlotter(
            replicator_func=lambda x, t: replicator_dyn(x, t, payoff_matrix),
            strategy_labels=strategy_labels,
            corner_label_fontsize=corner_label_fontsize
        )
        plotter.plot_dynamics_simplex(ax, colorbar_label_fontsize=colorbar_label_fontsize, show_colorbar_label=show_colorbar_label)
    except Exception as e:
        print(f"Error during simplex dynamics plotting: {e}"); traceback.print_exc(); plt.close(fig); return

    if show_trajectory and population_history_data is not None and plotter is not None:
        traj_ba = population_history_data
        traj_ba_sum = np.sum(traj_ba, axis=1, keepdims=True)
        traj_ba_normalized = np.divide(traj_ba, traj_ba_sum,
                                       out=np.full_like(traj_ba, 1/3.),
                                       where=traj_ba_sum != 0)
        traj_xy = plotter.ba2xy(traj_ba_normalized)
        x_coords, y_coords = traj_xy[:, 0], traj_xy[:, 1]
        ax.plot(x_coords, y_coords, c='magenta', lw=trajectory_line_width, ls='-',
                marker='.', ms=trajectory_marker_size, label='Moran Trajectory', zorder=4)
        if len(x_coords) > 0:
            ax.plot(x_coords[0], y_coords[0], 'o', c='lime', ms=start_end_marker_size,
                    label='Start', zorder=6, mec='k')
            ax.plot(x_coords[-1], y_coords[-1], 's', c='red', ms=start_end_marker_size,
                    label=f'End (Step {len(x_coords)-1})', zorder=6, mec='k')
    else:
        print("  Moran trajectory will not be plotted.")

    handles, legend_labels_list = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(legend_labels_list, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left',
                  bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=legend_fontsize)
    
    fig.tight_layout(rect=[0, 0, 0.82, 0.92])
    
    safe_run_name = re.sub(r'[^\w\-]+', '_', tournament_run_name)
    # Add trajectory status to filename
    if show_trajectory:
        traj_suffix = "with_trajectory"
    else:
        traj_suffix = "ReplicatorDynamics"
    outfile = output_dir / f"Simplex_{safe_run_name}.pdf"
    try:
        plt.savefig(outfile, dpi=150, bbox_inches='tight'); plt.close(fig)
        print(f"  Saved Simplex plot to: {outfile}")
    except Exception as e:
        print(f"Error saving plot {outfile}: {e}"); plt.close(fig)

def plot_moran_trajectories_on_simplex(
    history_paths: List[Path],
    strategy_labels: List[str],
    output_dir: Path,
    tournament_run_name: str,
    corner_label_fontsize: int = 28,
    main_title_fontsize: int = 30,
    figsize: tuple = SIMPLEX_FIGSIZE
    ):
    """
    Plots multiple Moran process trajectories on a blank simplex.
    """
    print(f"  Generating combined Moran Trajectories plot for {len(history_paths)} seeds...")
    if len(strategy_labels) != 3:
        print("Error: Simplex plot requires exactly 3 strategies.")
        return

    fig, ax = plt.subplots(figsize=figsize)
    
    # 1. Draw the blank simplex triangle
    ax.triplot(simplex_triangle, lw=0.8, c="darkgrey", zorder=1)

    # 2. Define a local barycentric-to-cartesian converter
    def ba2xy(ba):
        ba=np.array(ba)
        return simplex_corners.T.dot(ba.T).T if ba.ndim > 1 else simplex_corners.T.dot(ba)

    # 3. Plot each trajectory
    colors = plt.cm.jet(np.linspace(0, 1, len(history_paths)))
    for i, history_path in enumerate(history_paths):
        try:
            pop_history_df = pd.read_csv(history_path)
            if not all(label in pop_history_df.columns for label in strategy_labels):
                print(f"    Warning: Skipping {history_path}, missing columns.")
                continue
            
            traj_ba = pop_history_df[strategy_labels].to_numpy()
            traj_ba_sum = np.sum(traj_ba, axis=1, keepdims=True)
            traj_ba_normalized = np.divide(traj_ba, traj_ba_sum,
                                           out=np.full_like(traj_ba, 1/3.),
                                           where=traj_ba_sum != 0)
            traj_xy = ba2xy(traj_ba_normalized)
            x_coords, y_coords = traj_xy[:, 0], traj_xy[:, 1]
            
            # Plot the line with some transparency
            ax.plot(x_coords, y_coords, c=colors[i], lw=2.0, ls='-',
                    alpha=0.6, label=f'Seed {i+1}', zorder=4)
            
            # Plot start and end markers
            if len(x_coords) > 0:
                ax.plot(x_coords[0], y_coords[0], 'o', c=colors[i], ms=8, zorder=6, mec='k', alpha=0.7) # Start
                ax.plot(x_coords[-1], y_coords[-1], 's', c=colors[i], ms=8, zorder=6, mec='k', alpha=0.7) # End

        except Exception as e:
            print(f"    Warning: Could not plot trajectory for {history_path}. Error: {e}")

    # 4. Add corner labels
    mgn = 0.05
    ax.text(simplex_corners[0,0], simplex_corners[0,1]-mgn, strategy_labels[0], ha='center',va='top',fontsize=corner_label_fontsize,weight='bold')
    ax.text(simplex_corners[1,0], simplex_corners[1,1]-mgn, strategy_labels[1], ha='center',va='top',fontsize=corner_label_fontsize,weight='bold')
    ax.text(simplex_corners[2,0], simplex_corners[2,1]+mgn*0.5, strategy_labels[2], ha='center',va='bottom',fontsize=corner_label_fontsize,weight='bold')

    # 5. Finalize and save
    ax.axis('equal'); ax.axis('off')
    ax.set_ylim(ymin=-0.1, ymax=simplex_corners[2,1]+0.1); ax.set_xlim(xmin=-0.1, xmax=1.1)
    
    if len(history_paths) <= 15: # Add a legend if it's not too crowded
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=LEGEND_FONTSIZE)
    
    plt.title("Moran Process Trajectories (All Seeds)", fontsize=main_title_fontsize)
    fig.tight_layout(rect=[0, 0, 0.82, 0.92])

    safe_run_name = re.sub(r'[^\w\-]+', '_', tournament_run_name)
    outfile = output_dir / f"Simplex_Plot_{safe_run_name}_MoranTrajectories.pdf"
    try:
        plt.savefig(outfile, dpi=150, bbox_inches='tight'); plt.close(fig)
        print(f"  Saved combined Moran Trajectories plot to: {outfile}")
    except Exception as e:
        print(f"Error saving Moran plot {outfile}: {e}"); plt.close(fig)

def plot_simplex_panel_from_dyadic_data(
    coin_game_matrix_path: Path,
    ipd_matrix_path: Path,
    output_dir: Path,
    corner_label_fontsize: int = 32,
    colorbar_label_fontsize: int = 28,
    main_title_fontsize: int = 34
    ):
    """
    Creates a side-by-side simplex panel with Coin Game and IPD plots.
    Both simplexes will have exactly the same size.
    """
    print(f"\n--- Generating Side-by-Side Simplex Panel ---")
    print(f"  Coin Game matrix: {coin_game_matrix_path}")
    print(f"  IPD matrix: {ipd_matrix_path}")
    
    # Load payoff matrices
    try:
        coin_game_df = pd.read_csv(coin_game_matrix_path, index_col=0)
        ipd_df = pd.read_csv(ipd_matrix_path, index_col=0)
        
        if len(coin_game_df.columns) != 3 or coin_game_df.shape != (3,3):
            print(f"Error: Coin Game matrix must have 3x3 shape. Found {coin_game_df.shape}")
            return
        if len(ipd_df.columns) != 3 or ipd_df.shape != (3,3):
            print(f"Error: IPD matrix must have 3x3 shape. Found {ipd_df.shape}")
            return
            
        coin_game_matrix = coin_game_df.to_numpy()
        ipd_matrix = ipd_df.to_numpy()
        strategy_labels = list(coin_game_df.columns)
        
    except Exception as e:
        print(f"Error loading payoff matrices: {e}")
        return
    
    # Create the panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SIMPLEX_PANEL_FIGSIZE)
    
    # Define replicator dynamics function
    def replicator_dyn(x_props, t, A_matrix):
        x_props = np.clip(np.array(x_props), 0, 1)
        x_sum = np.sum(x_props)
        x_props = x_props / x_sum if x_sum > 1e-9 else np.full_like(x_props, 1/len(x_props))
        expected_payoffs = A_matrix.dot(x_props)
        average_population_payoff = x_props.dot(expected_payoffs)
        return x_props * (expected_payoffs - average_population_payoff)
    
    # Plot Coin Game simplex (left)
    try:
        plotter1 = SimplexDynamicsPlotter(
            replicator_func=lambda x, t: replicator_dyn(x, t, coin_game_matrix),
            strategy_labels=strategy_labels,
            corner_label_fontsize=corner_label_fontsize
        )
        plotter1.plot_dynamics_simplex(ax1, colorbar_label_fontsize=colorbar_label_fontsize, show_colorbar_label=True)
        
        # Add title for Coin Game
        ax1.text(0.08, 0.5, "Coin Game", transform=ax1.transAxes, 
                fontsize=main_title_fontsize, va='center', ha='right', 
                rotation=90, fontfamily='monospace', weight='bold')
    except Exception as e:
        print(f"Error plotting Coin Game simplex: {e}")
        ax1.text(0.5, 0.5, "Error in Coin Game\nSimplex Generation", 
                transform=ax1.transAxes, ha='center', va='center')
    
    # Plot IPD simplex (right)
    try:
        plotter2 = SimplexDynamicsPlotter(
            replicator_func=lambda x, t: replicator_dyn(x, t, ipd_matrix),
            strategy_labels=strategy_labels,
            corner_label_fontsize=corner_label_fontsize
        )
        plotter2.plot_dynamics_simplex(ax2, colorbar_label_fontsize=colorbar_label_fontsize, show_colorbar_label=True)
        
        # Add title for IPD
        ax2.text(0.08, 0.5, "IPD", transform=ax2.transAxes, 
                fontsize=main_title_fontsize, va='center', ha='right', 
                rotation=90, fontfamily='monospace', weight='bold')
    except Exception as e:
        print(f"Error plotting IPD simplex: {e}")
        ax2.text(0.5, 0.5, "Error in IPD\nSimplex Generation", 
                transform=ax2.transAxes, ha='center', va='center')
    
    # Save the panel
    output_file = output_dir / "Simplex_Panel_CoinGame_IPD.pdf"
    try:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved simplex panel to: {output_file}")
    except Exception as e:
        print(f"Error saving panel {output_file}: {e}")
        plt.close(fig)

def generate_payoff_matrices(all_payoff_data: pd.DataFrame, output_dir: Path):
    """
    Computes and saves payoff matrices from aggregated dyadic run data.
    Matrix[row, col] = Payoff of 'row' agent when playing 'col' agent.
    """
    print(f"\n{'='*20} GENERATING PAYOFF MATRICES {'='*20}")
    
    # 1. Calculate the mean payoff for each agent in each run (averaging across meta-rounds)
    grouping_vars = ['game', 'pairing', 'seed', 'acting_agent', 'opponent_agent']
    payoffs_per_run = all_payoff_data.groupby(grouping_vars)['payoff'].mean().reset_index()

    # 2. Calculate the final mean and SEM across all seed runs.
    final_stats = payoffs_per_run.groupby(['game', 'acting_agent', 'opponent_agent'])['payoff'].agg(['mean', 'sem'])

    # 3. Get unique games and agents
    unique_games = all_payoff_data['game'].unique()
    # Use 'acting_agent' to get the list of all agents
    agent_groups = sorted(list(all_payoff_data['acting_agent'].unique()))
    print(f"Found games: {unique_games}")
    print(f"Found agents: {agent_groups}")

    if not agent_groups:
        print("No agent data found, cannot generate matrices.")
        return

    # 4. Create a matrix for each game
    for game in unique_games:
        print(f"\n--- Payoff Matrix for: {game} ---")
        # Initialize an empty matrix
        payoff_matrix = pd.DataFrame(index=agent_groups, columns=agent_groups, dtype=float)
        
        for row_agent in agent_groups: # The "Actor"
            for col_agent in agent_groups: # The "Opponent"
                try:
                    # Get the mean payoff of row_agent playing col_agent
                    mean_payoff = final_stats.loc[(game, row_agent, col_agent)]['mean']
                    payoff_matrix.loc[row_agent, col_agent] = mean_payoff
                except KeyError:
                    # This pairing (e.g., PM vs PM) might not exist in the data
                    payoff_matrix.loc[row_agent, col_agent] = np.nan

        print("Payoff for 'row' agent vs 'column' agent:")
        print(payoff_matrix.to_string(float_format="%.4f"))
        
        # Save to CSV
        matrix_path = output_dir / f"payoff_matrix_{game}.csv"
        payoff_matrix.to_csv(matrix_path)
        print(f"Saved matrix to: {matrix_path}")
        
    print(f"\n{'='*20} PAYOFF MATRICES COMPLETE {'='*20}")


def analyze_moran_process(moran_histories: List[pd.DataFrame], output_dir: Path):
    """Analyzes and plots Moran process population dynamics."""
    if not moran_histories:
        print("No Moran process data to analyze.")
        return

    # Align columns and concatenate
    all_history_df = pd.concat([df.set_index('moran_step') for df in moran_histories], axis=1, keys=range(len(moran_histories)))
    
    mean_df = all_history_df.groupby(level=1, axis=1).mean()
    sem_df = all_history_df.groupby(level=1, axis=1).sem()

    plt.figure(figsize=(12, 7))
    for strategy in mean_df.columns:
        mean_vals = mean_df[strategy]
        sem_vals = sem_df[strategy]
        
        plt.plot(mean_df.index, mean_vals, label=strategy)
        plt.fill_between(mean_df.index, mean_vals - sem_vals, mean_vals + sem_vals, alpha=0.2)

    plt.title("Moran Process Population Dynamics", fontsize=TITLE_FONTSIZE)
    plt.xlabel("Moran Step", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Population Fraction", fontsize=AXIS_LABEL_FONTSIZE)
    plt.legend(title="Strategy", fontsize=LEGEND_FONTSIZE)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 1)

    plot_path = output_dir / "evolutionary_moran_dynamics.pdf"
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    plt.close()

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(
        description="Analysis suite for multi-agent game simulation results.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--results_dir", type=Path, required=True,
        help="The base directory containing all your 'results_*' game folders."
    )
    parser.add_argument(
        "--individual", action="store_true",
        help="If set, runs detailed analysis on every pairing within every 'results_*' folder."
    )
    parser.add_argument(
        "--aggregate", action="store_true",
        help="If set, performs an aggregated analysis for each game, storing results in separate folders."
    )
    parser.add_argument(
        "--games", type=str, nargs='+',
        help="Optional: Specify one or more game names to run the analysis on (e.g., CoinGame IPD)."
    )
    args = parser.parse_args()

    if not args.individual and not args.aggregate:
        parser.error("No analysis selected. Please specify --individual and/or --aggregate.")

    if args.individual:
        print(f"\n{'='*20} STARTING INDIVIDUAL ANALYSIS {'='*20}")
        
        top_level_dirs = [d for d in args.results_dir.iterdir() if d.is_dir() and d.name.startswith('results_')]
        
        if args.games:
            print(f"Filtering for games: {args.games}")
            top_level_dirs = [d for d in top_level_dirs if any(game_name in d.name for game_name in args.games)]

        if not top_level_dirs:
             print("No 'results_*' directories found for individual analysis.")
        else:
            print(f"Found {len(top_level_dirs)} top-level game directories to analyze.")

        for game_dir in top_level_dirs:
            print(f"\n--- Analyzing Game: {game_dir.name} ---")
            pairing_dirs = [d for d in game_dir.iterdir() if d.is_dir() and not d.name.endswith('_plots')]
            if not pairing_dirs:
                print(f"No pairing folders found in {game_dir.name}. Skipping.")
                continue

            for pairing_dir in pairing_dirs:
                print(f"\n--- Processing Pairing: {pairing_dir.name} ---")
                output_dir = pairing_dir / "analysis_plots"
                output_dir.mkdir(parents=True, exist_ok=True)
                print(f"Individual analysis outputs will be saved to: {output_dir}")

                openai_client = load_api_key()

                # --- START FIX ---
                # 1. Check for EVOLUTIONARY data at the PAIRING level FIRST
                payoff_path = pairing_dir / "payoff_matrix.csv"
                moran_path = pairing_dir / "moran_process_history.csv"

                if payoff_path.exists():
                    print("\n--- Found Evolutionary Tournament Data ---")
                    try:
                        tournament_run_name = pairing_dir.name
                        
                        # Plot 1: Replicator Dynamics (Flow field from AVG matrix, NO trajectories)
                        print("  Generating Replicator Dynamics simplex plot (no trajectories)...")
                     
                        game_name = pairing_dir.parent.name.replace('results_', '')
                        title = game_name
                        plot_simplex_from_tournament_data(
                            payoff_matrix_path=payoff_path,
                            population_history_path=moran_path, # Not used, but required by func
                            output_dir=output_dir,
                            tournament_run_name=tournament_run_name,
                            plot_title=title,
                            show_trajectory=False
                        )

                        # Plot 2: Moran Trajectories (BLANK simplex, ALL trajectories)
                        print("  Generating Moran Trajectories simplex plot...")
                        # Find ALL seed history files
                        history_paths = list(pairing_dir.glob("moran_process_history_seed_*.csv"))
                        
                        if history_paths:
                            # Get labels from the payoff matrix
                            try:
                                strategy_labels = list(pd.read_csv(payoff_path, index_col=0).columns)
                                plot_moran_trajectories_on_simplex(
                                    history_paths=history_paths,
                                    strategy_labels=strategy_labels,
                                    output_dir=output_dir,
                                    tournament_run_name=tournament_run_name
                                )
                            except Exception as e:
                                print(f"    Failed to get strategy labels for Moran plot: {e}")
                        else:
                            print("    No moran_process_history_seed_*.csv files found. Skipping trajectory plot.")
                        
                    except Exception as e:
                        print(f"Error processing evolutionary data in {pairing_dir}: {e}")
                        traceback.print_exc()

                # 2. If not evolutionary, process as DYADIC data by looking at run_dirs
                else:
                    print("\n--- No evolutionary data found, checking for Dyadic Meta-Game Data ---")
                    run_dirs = find_experiment_runs(pairing_dir)
                    if not run_dirs:
                        print("No valid run (seed) directories found. Skipping.")
                        continue
                        
                    all_dyadic_data = []
                    for run_dir in run_dirs:
                        processed_data = process_dyadic_run(run_dir)
                        if processed_data:
                            all_dyadic_data.append(processed_data['data'])
                    
                    if all_dyadic_data:
                        print("\n--- Running Dyadic Meta-Game Analysis ---")
                        analyze_dyadic_metrics(all_dyadic_data, output_dir)
                        run_llm_as_judge(all_dyadic_data, output_dir, openai_client)

                        all_log_dfs = []
                        for run_dir in run_dirs:
                            log_file = next(run_dir.glob("log_seed_*.csv"), None)
                            if log_file and log_file.exists():
                                all_log_dfs.append(pd.read_csv(log_file))
                        if all_log_dfs:
                            plot_logged_payoffs_over_time(all_log_dfs, output_dir)
                    else:
                        print("Found no dyadic data to process in this pairing.")
        print(f"\n{'='*20} INDIVIDUAL ANALYSIS COMPLETE {'='*20}")

    if args.aggregate:
        print(f"\n{'='*20} STARTING AGGREGATED ANALYSIS {'='*20}")
        
        # 1. Define main output directory
        aggregated_output_dir = args.results_dir / "aggregated_analysis_plots"
        aggregated_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Aggregated outputs will be saved to: {aggregated_output_dir}")

        # 2. Calculate metrics across ALL specified games at once to get the complete dataset
        mean_df_all, sem_df_all, per_experiment_data_all = average_radon_metrics_across_experiments(args.results_dir, games_to_run=args.games)

        payoff_mean_all, payoff_sem_all, payoff_per_exp_all = average_payoffs_across_experiments(args.results_dir, games_to_run=args.games)

        if per_experiment_data_all is not None and not per_experiment_data_all.empty:
            # 3. Generate and save the main panel plot containing all games
            plot_aggregated_metrics_panel(mean_df_all, sem_df_all, per_experiment_data_all, aggregated_output_dir)

            # 4. Loop through each unique game to create individual plots in dedicated subdirectories
            exp_names = per_experiment_data_all.index.get_level_values('experiment').unique()
            unique_games = sorted(list(set([re.search(r'results_([^/]+)', name).group(1) for name in exp_names if re.search(r'results_([^/]+)', name)])))
            
            print(f"\n--- Generating individual aggregated plots for each game: {unique_games} ---")
            for game_name in unique_games:
                # Create a specific subdirectory for this game
                game_subdir = aggregated_output_dir / f"{game_name}_aggregated_metrics"
                print(f"  - Preparing plots for {game_name} in: {game_subdir}")

                # Filter the complete dataset for the current game
                per_experiment_data_game = per_experiment_data_all[per_experiment_data_all.index.get_level_values('experiment').str.contains(f"results_{game_name}")]
                
                # Recalculate mean and sem for just this game's data
                metrics_to_average = per_experiment_data_game.columns.tolist()
                grouped_game = per_experiment_data_game.groupby('meta_round')
                mean_df_game = grouped_game[metrics_to_average].mean()
                sem_df_game = grouped_game[metrics_to_average].sem()

                # Call the new function to generate and save the individual plots in the subfolder
                generate_individual_metric_plots(mean_df_game, sem_df_game, per_experiment_data_game, game_subdir)

                # --- NEW SECTION for individual payoff plots ---
                if payoff_per_exp_all is not None and not payoff_per_exp_all.empty:
                    print(f"  - Preparing payoff plot for {game_name} in: {game_subdir}")
                    # Filter the payoff data for the current game
                    payoff_per_exp_game = payoff_per_exp_all[payoff_per_exp_all.index.get_level_values('experiment').str.contains(f"results_{game_name}")]
                    
                    if not payoff_per_exp_game.empty:
                        # Recalculate mean and sem for just this game's payoffs
                        grouped_payoff_game = payoff_per_exp_game.groupby('meta_round')
                        mean_df_payoff_game = grouped_payoff_game[['payoff']].mean()
                        sem_df_payoff_game = grouped_payoff_game[['payoff']].sem()
                        
                        # Call the new function to generate the plot
                        generate_individual_payoff_plot(mean_df_payoff_game, sem_df_payoff_game, payoff_per_exp_game, game_subdir)
                    else:
                        print(f"    - No payoff data found for {game_name}.")
        else:
            print("No data found to generate aggregated plots.")
        
        all_llm_data = load_all_llm_judge_data(args.results_dir, games_to_run=args.games)
        if all_llm_data is not None and not all_llm_data.empty:
            generate_strategy_proportion_table(all_llm_data, aggregated_output_dir)
            plot_aggregated_strategy_panel(all_llm_data, aggregated_output_dir)
        else:
            print("No LLM data found to generate aggregated strategy proportion plot or table.")

        print("\n--- Generating Aggregated Payoff Bar Plot (by Agent Type) ---")
        all_payoff_data = load_all_payoff_data_by_agent(args.results_dir, games_to_run=args.games)
        
        if all_payoff_data is not None and not all_payoff_data.empty:
            plot_aggregated_payoff_bars(all_payoff_data, aggregated_output_dir)
            generate_payoff_matrices(all_payoff_data, aggregated_output_dir)

            # --- NEW: Generate Simplex Plots from Dyadic Payoff Matrices ---
            print("\n--- Generating Replicator Dynamics Simplex Plots (from Aggregated Dyadic Data) ---")
            
            # We can only generate a simplex if there are exactly 3 agent types
            agent_groups = sorted(list(all_payoff_data['acting_agent'].unique()))
            if len(agent_groups) != 3:
                print(f"Skipping simplex plot generation: Requires exactly 3 agent types, but found {len(agent_groups)} ({agent_groups}).")
            else:
                unique_games = all_payoff_data['game'].unique()
                for game in unique_games:
                    matrix_path = aggregated_output_dir / f"payoff_matrix_{game}.csv"
                    
                    if matrix_path.exists():
                        print(f"  Generating simplex plot for {game}...")
                        dummy_history_path = matrix_path 
                        show_cbar_label = False if game == "CoinGame" else True

                        title = f"{game}".replace("CoinGame", "Coin Game")
                        plot_simplex_from_tournament_data(
                            payoff_matrix_path=matrix_path,
                            population_history_path=dummy_history_path,
                            output_dir=aggregated_output_dir,
                            tournament_run_name=f"{game}_from_Dyadic",
                            plot_title=title,
                            show_colorbar_label=show_cbar_label,
                            show_trajectory=False
                        )
                    else:
                        print(f"  Skipping simplex for {game}: Payoff matrix not found at {matrix_path}")
                
                # Generate side-by-side simplex panel if both games are available
                coin_game_matrix_path = aggregated_output_dir / "payoff_matrix_CoinGame.csv"
                ipd_matrix_path = aggregated_output_dir / "payoff_matrix_IPD.csv"
                
                if coin_game_matrix_path.exists() and ipd_matrix_path.exists():
                    print(f"\n  Generating side-by-side simplex panel...")
                    plot_simplex_panel_from_dyadic_data(
                        coin_game_matrix_path=coin_game_matrix_path,
                        ipd_matrix_path=ipd_matrix_path,
                        output_dir=aggregated_output_dir
                    )
                else:
                    print(f"  Skipping simplex panel: Missing payoff matrices")
                    if not coin_game_matrix_path.exists():
                        print(f"    Missing: {coin_game_matrix_path}")
                    if not ipd_matrix_path.exists():
                        print(f"    Missing: {ipd_matrix_path}")

        else:
            print("No agent-specific payoff data found to generate aggregated bar plot.")

        print(f"\n{'='*20} AGGREGATED ANALYSIS COMPLETE {'='*20}")

    print(f"\n{'='*20} ALL REQUESTED ANALYSES COMPLETE {'='*20}")

if __name__ == "__main__":
    main()