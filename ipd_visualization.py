#%%
"""
IPD Visualization - Visualizes results from the Iterated Prisoner's Dilemma LLM analysis.
"""

import os
import glob
import traceback
import re
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import ast
# ADDED: Import for t-tests
from scipy.stats import ttest_ind


try:
    import axelrod
    AXELROD_AVAILABLE = True
except ImportError:
    print("WARNING: Axelrod library not found. Cannot perform check against axelrod.all_strategies.")
    AXELROD_AVAILABLE = False

try:
    import dataframe_image as dfi
    HAS_DFI = True
except ImportError:
    HAS_DFI = False

# --- Configuration ---
DEFAULT_MODEL_CONFIGS = [
    ## First run: Obfuscated runs for both of these models
    {
        "name": "deepseek-ai/DeepSeek-R1", 
        "api": "huggingface", 
        "display_name": "DeepSeek-R1", 
        "source_type": "reasoning" 
    },
    {
        "name": "mistralai/Mistral-Small-24B-Instruct-2501",
        "api": "huggingface",
        "display_name": "Mistral Small (24B) (Instruct)",
        "source_type": "open"
    },
    {
       "name": "Qwen/Qwen2.5-7B-Instruct",
       "api": "huggingface",
       "display_name": "Qwen 2.5 (7B) (Instruct)",
       "source_type": "open"
    },
    {
        "name": "Qwen/Qwen2.5-72B-Instruct",
        "api": "huggingface",
        "display_name": "Qwen 2.5 (72B) (Instruct)",
        "source_type": "open"
    },
    {
        "name": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "api": "huggingface",
        "display_name": "Qwen 2.5 Coder (32B) (Instruct)",
        "source_type": "open"
    },
    {
       "name": "gpt-4o-mini-2024-07-18",
       "api": "openai",
       "display_name": "GPT-4o Mini",
       "source_type": "closed"
    },
    {
        "name": "gpt-4.1-mini-2025-04-14", 
        "api": "openai",
        "display_name": "GPT-4.1 Mini",
        "source_type": "closed"
    },
    {
        "name": "gpt-4.1-nano-2025-04-14",
        "api": "openai",
        "display_name": "GPT-4.1 Nano",
        "source_type": "closed"
    },
    {
        "name": "o4-mini-2025-04-16",
        "api": "openai",
        "display_name": "o4-mini",
        "source_type": "reasoning"
    },
    {
        "name": "gpt-4.1-2025-04-14",
        "api": "openai",
        "display_name": "GPT-4.1",
        "source_type": "closed"
    },
    {
        "name": "deepseek-ai/DeepSeek-V3-0324",
        "api": "huggingface",
        "display_name": "DeepSeek-V3",
        "source_type": "open"
    }
]

# --- Manual Overrides for specific files ---
MANUAL_OVERRIDES = {
    # Filename: is_stochastic (True/False)
    "NMWEDeterministic.py": False,
    "NMWEFiniteMemory.py": True,
    "NMWELongMemory.py": True,
    "NMWEMemoryOne.py": True,
    "NMWEStochastic.py": True,
}
print(f"Defined manual overrides for {len(MANUAL_OVERRIDES)} files.")

# --- Summary Table Functions --- 

def count_loc(filepath):
    """Counts lines of code, excluding comments and blank lines."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        loc = 0
        for line in lines:
            stripped_line = line.strip()
            # Count lines that are not empty and not starting with #
            if stripped_line and not stripped_line.startswith('#'):
                loc += 1
        return loc
    except Exception as e:
        print(f"Error reading file {filepath} for LOC: {e}")
        return None

def parse_classifier_info(filepath):
    """
    Parses a Python file using AST to find 'stochastic' and 'memory_depth'
    within the first 'classifier' dictionary found in any class.
    """
    result = {'stochastic': None, 'memory_depth': None, 'status': 'no_classifier'}
    classifier_assignment_found = False
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            file_content = f.read()
        tree = ast.parse(file_content, filename=filepath)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for class_node in node.body:
                    if (isinstance(class_node, ast.Assign) and
                            len(class_node.targets) == 1 and
                            isinstance(class_node.targets[0], ast.Name) and
                            class_node.targets[0].id == 'classifier'):

                        classifier_assignment_found = True
                        if isinstance(class_node.value, ast.Dict):
                            classifier_dict = class_node.value
                            stochastic_found = False
                            memory_depth_found = False

                            for i, key_node in enumerate(classifier_dict.keys):
                                if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                                    continue # Skip non-string keys

                                key_name = key_node.value
                                value_node = classifier_dict.values[i]

                                # --- Parse Stochastic ---
                                if key_name == 'stochastic':
                                    stochastic_found = True
                                    if isinstance(value_node, ast.Constant) and isinstance(value_node.value, bool):
                                        result['stochastic'] = value_node.value
                                    else:
                                        print(f"Warning: Invalid value for 'stochastic' in {os.path.basename(filepath)}")
                                        result['status'] = 'invalid_stochastic'
                                        # Keep going to check memory depth

                                # --- Parse Memory Depth ---
                                elif key_name == 'memory_depth':
                                    memory_depth_found = True
                                    # Case 1: Integer literal (e.g., 0, 1, 5)
                                    if isinstance(value_node, ast.Constant) and isinstance(value_node.value, int):
                                        result['memory_depth'] = value_node.value
                                    # Case 2: float('inf')
                                    elif (isinstance(value_node, ast.Call) and
                                            isinstance(value_node.func, ast.Name) and
                                            value_node.func.id == 'float' and
                                            len(value_node.args) == 1 and
                                            isinstance(value_node.args[0], ast.Constant) and
                                            value_node.args[0].value == 'inf'):
                                        result['memory_depth'] = float('inf') # Use actual infinity
                                    else:
                                        print(f"Warning: Unrecognized value for 'memory_depth' in {os.path.basename(filepath)}")
                                        result['status'] = 'invalid_memory'
                                        # Keep going to check stochastic

                            # --- Determine overall status after checking keys ---
                            if result['status'] not in ['invalid_stochastic', 'invalid_memory']: # If no specific value error occurred
                                if stochastic_found and memory_depth_found:
                                    result['status'] = 'ok'
                                elif not stochastic_found and memory_depth_found:
                                    result['status'] = 'missing_stochastic'
                                elif stochastic_found and not memory_depth_found:
                                    result['status'] = 'missing_memory'
                                else: # Neither found
                                    result['status'] = 'missing_both' # More specific status

                            return result 

                        else: 
                            print(f"Warning: Found 'classifier' assignment in {os.path.basename(filepath)}, but it's not a direct dictionary literal.")
                            result['status'] = 'invalid_classifier_format'
                            return result 

        if not classifier_assignment_found:
            result['status'] = 'no_classifier'

        return result

    except SyntaxError as e:
        print(f"Syntax Error parsing {os.path.basename(filepath)}: {e}")
        return {'stochastic': None, 'memory_depth': None, 'status': 'parse_error'}
    except Exception as e:
        print(f"General Error parsing {os.path.basename(filepath)} with AST: {e}")
        traceback.print_exc()
        return {'stochastic': None, 'memory_depth': None, 'status': 'parse_error'}


def get_strategy_stats_from_files(strategies_dir):
    """
    Analyzes Python files in a directory using AST parsing
    to extract strategy statistics safely without execution.
    Compares found files against axelrod.all_strategies if available.
    Applies manual overrides for stochastic status for specified files.

    Args:
        strategies_dir (str): Path to the directory containing strategy .py files.

    Returns:
        dict: A dictionary containing statistics like counts, LOC, etc.
                Returns None if the directory is invalid or Axelrod unavailable when needed.
    """
    if not os.path.isdir(strategies_dir):
        print(f"Error: Directory not found: {strategies_dir}")
        return None

    # --- Axelrod Check ---
    official_strategy_classes = set()
    if AXELROD_AVAILABLE:
        try:
            official_strategy_classes = {strat.__name__ for strat in axelrod.all_strategies}
            print(f"Successfully loaded {len(official_strategy_classes)} strategy names from Axelrod library.")
        except Exception as e:
            print(f"Error accessing axelrod.all_strategies: {e}. Proceeding without Axelrod list check.")
            official_strategy_classes = set()
    else:
        print("Skipping check against axelrod.all_strategies.")
    # --- End Axelrod Check ---

    py_files = glob.glob(os.path.join(strategies_dir, '*.py'))
    print(f"Found {len(py_files)} Python files in {strategies_dir}.")

    stats = {
        'total_files_scanned': 0,
        'files_matching_axelrod': 0,
        'strategies_analyzed': 0, # Files matching Axelrod where classifier info was obtained (parsed or manual)
        'stochastic_count': 0,
        'deterministic_count': 0,
        'memory_depth_counts': {0: 0, 1: 0, 'finite_gt_1': 0, float('inf'): 0, 'unknown': 0},
        'files_with_issues': [], # Store tuples: (filename, issue_status) for non-overridden files
        'files_manually_classified': [], # Store filenames that used override for stochastic
        'loc_counts_matched': [], # LOC for files matching Axelrod list
        'files_not_in_axelrod': [], # Store filenames not in official list
        'parse_read_errors': 0, # Count files that failed basic parsing/reading for LOC or AST
    }

    all_filenames_found = set()

    for filepath in py_files:
        filename = os.path.basename(filepath)
        all_filenames_found.add(filename)

        if filename == "__init__.py":
            continue

        stats['total_files_scanned'] += 1
        strategy_name = filename[:-3]

        is_official = strategy_name in official_strategy_classes
        if not is_official and AXELROD_AVAILABLE and official_strategy_classes:
            stats['files_not_in_axelrod'].append(filename)
            continue

        stats['files_matching_axelrod'] += 1

        # --- Initialize results for this file ---
        stochastic_flag = None
        memory_depth = None
        status = 'unknown' # Default status before processing
        is_manual_override = False

        # --- Get LOC first ---
        loc = count_loc(filepath)
        if loc is None:
            stats['parse_read_errors'] += 1
            status = 'loc_read_error'
            stats['files_with_issues'].append((filename, status))
        else:
            stats['loc_counts_matched'].append(loc) # Add LOC only if successful


        # --- Check for Manual Override (Stochastic only) ---
        if filename in MANUAL_OVERRIDES:
            stochastic_flag = MANUAL_OVERRIDES[filename]
            status = 'manual_override_stochastic' # Specific status
            is_manual_override = True
            stats['files_manually_classified'].append(filename)
            print(f"      - Applying manual override for {filename}: stochastic={stochastic_flag}")
            # Attempt to parse memory depth even if stochastic is overridden
            parsed_info = parse_classifier_info(filepath)
            memory_depth = parsed_info['memory_depth']
            if parsed_info['status'] == 'parse_error' and status != 'loc_read_error':
                stats['parse_read_errors'] += 1
                stats['files_with_issues'].append((filename, 'parse_error_override')) # Special issue code
            elif memory_depth is None and parsed_info['status'] not in ['parse_error', 'no_classifier', 'invalid_classifier_format']:
                stats['files_with_issues'].append((filename, 'missing_memory_override'))
        else:
            parsed_info = parse_classifier_info(filepath)
            stochastic_flag = parsed_info['stochastic']
            memory_depth = parsed_info['memory_depth']
            status = parsed_info['status'] 
        # --- End Override/Parsing ---


        # --- Tally Results ---
        if status in ['ok', 'manual_override_stochastic', 'missing_stochastic', 'missing_memory', 'missing_both']:
            stats['strategies_analyzed'] += 1

            if stochastic_flag is True:
                stats['stochastic_count'] += 1
            elif stochastic_flag is False:
                stats['deterministic_count'] += 1
            if isinstance(memory_depth, int):
                if memory_depth == 0:
                    stats['memory_depth_counts'][0] += 1
                elif memory_depth == 1:
                    stats['memory_depth_counts'][1] += 1
                elif memory_depth > 1:
                    stats['memory_depth_counts']['finite_gt_1'] += 1
            elif memory_depth == float('inf'):
                stats['memory_depth_counts'][float('inf')] += 1
            else: # memory_depth is None or invalid
                stats['memory_depth_counts']['unknown'] += 1

            if status not in ['ok', 'manual_override_stochastic']:
                if not any(f == filename and s.startswith('loc_read_error') for f,s in stats['files_with_issues']):
                    stats['files_with_issues'].append((filename, status))

        elif status not in ['loc_read_error']: 
            if status == 'parse_error':
                if loc is not None: stats['parse_read_errors'] += 1
            if not any(f == filename for f,s in stats['files_with_issues']): 
                stats['files_with_issues'].append((filename, status))
        # --- End Tally Results ---


    # Calculate final stats
    if stats['loc_counts_matched']:
        stats['min_loc'] = min(stats['loc_counts_matched'])
        stats['max_loc'] = max(stats['loc_counts_matched'])
        stats['avg_loc'] = np.mean(stats['loc_counts_matched'])
    else:
        stats['min_loc'] = 0
        stats['max_loc'] = 0
        stats['avg_loc'] = 0.0

    # --- Print Detailed File Lists ---
    print(f"\nAnalysis complete.")
    print(f" -> Analyzed {stats['strategies_analyzed']} strategies (matching Axelrod list or check skipped) where classifier info could be partially or fully obtained.")

    if stats['files_manually_classified']:
        print(f" -> Applied manual stochastic classification for {len(stats['files_manually_classified'])} files.")

    if stats['files_with_issues']:
        print(f" -> Found {len(stats['files_with_issues'])} files matching Axelrod list (or check skipped) with parsing/classifier issues (see details below).")
        # for fname, issue in stats['f  iles_with_issues']:
        #     print(f"    - {fname}: {issue}")
    if stats['parse_read_errors'] > 0:
         print(f" -> Encountered {stats['parse_read_errors']} files that could not be read or parsed for LOC/AST.")

    if AXELROD_AVAILABLE and official_strategy_classes and stats['files_not_in_axelrod']:
        print(f" -> Found {len(stats['files_not_in_axelrod'])} .py files in the directory that are NOT in axelrod.all_strategies.")
        # for fname in stats['files_not_in_axelrod']:
        #     print(f"    - {fname}")
    elif not official_strategy_classes and AXELROD_AVAILABLE:
        print(" -> Could not load Axelrod strategy list for comparison, so all files were processed.")


    return stats


def generate_strategy_stats_table(strategies_dir, output_dir):
    """
    Generates a LaTeX table summarizing strategy statistics from Python files,
    using safe AST parsing, checking against axelrod.all_strategies,
    applying manual overrides for stochastic status, and including memory depth.

    Args:
        strategies_dir (str): Path to the directory containing strategy .py files.
        output_dir (str): Directory to save the LaTeX file.
    """
    print(f"\n--- Generating Strategy Statistics Table (AST + Axelrod Check + Manual Overrides + Memory) from: {strategies_dir} ---")
    stats = get_strategy_stats_from_files(strategies_dir)
    if stats is None:
        print("Failed to get strategy statistics.")
        return

    os.makedirs(output_dir, exist_ok=True)
    latex_filepath = os.path.join(output_dir, "strategy_dataset_summary_detailed.tex") # New filename
    axelrod_check_performed = AXELROD_AVAILABLE and stats.get('files_not_in_axelrod') is not None
    footnote_symbol_loc = "§" if axelrod_check_performed else "*"
    footnote_symbol_issues = "†" if axelrod_check_performed else "§"


    latex_string = f"""\\documentclass{{article}}
\\usepackage{{booktabs}}
\\usepackage{{graphicx}} % Required for resizebox
\\usepackage{{caption}}  % Required for caption setup
\\captionsetup[table]{{skip=5pt}} % Add some space between caption and table

\\begin{{document}}

\\begin{{table}}[htbp]
\\centering
\\caption{{Summary of Axelrod Strategy Dataset Characteristics (Parsed from Python Files)}}
\\label{{tab:strategy_dataset_summary_detailed}}
\\resizebox{{\\textwidth}}{{!}}{{
\\begin{{tabular}}{{lc}}
\\toprule
Characteristic & Value \\\\
\\midrule
Total .py files scanned in directory & {stats['total_files_scanned']} \\\\
"""
    if axelrod_check_performed:
        latex_string += f"Files matching an Axelrod strategy name & {stats['files_matching_axelrod']} \\\\ \n"
        latex_string += f"Strategies analyzed (matched files where classifier info obtained) & {stats['strategies_analyzed']} \\\\ \n"
    else:
        latex_string += f"Strategies analyzed (classifier info obtained) & {stats['strategies_analyzed']} \\\\ \n"

    latex_string += f"""
Stochastic Strategies & {stats['stochastic_count']} \\\\
Deterministic Strategies & {stats['deterministic_count']} \\\\
Manual Stochastic Overrides Applied & {len(stats['files_manually_classified'])} \\\\
\\midrule
Memory Depth (for analyzed strategies): & \\\\
  Memory Depth 0 & {stats['memory_depth_counts'][0]} \\\\
  Memory Depth 1 & {stats['memory_depth_counts'][1]} \\\\
  Finite Memory Depth > 1 & {stats['memory_depth_counts']['finite_gt_1']} \\\\
  Infinite Memory Depth (float('inf')) & {stats['memory_depth_counts'][float('inf')]} \\\\
  Unknown/Unparsed Memory Depth & {stats['memory_depth_counts']['unknown']} \\\\
\\midrule
Lines of Code (LOC) for Matched/Analyzed Strategies$^{{{footnote_symbol_loc}}}$: & \\\\
  Minimum LOC & {stats['min_loc']} \\\\
  Maximum LOC & {stats['max_loc']} \\\\
  Average LOC & {stats['avg_loc']:.2f} \\\\
\\midrule
Files with parsing/classifier issues$^{{{footnote_symbol_issues}}}$ & {len(stats['files_with_issues'])} \\\\
Files unreadable/unparsable for LOC/AST & {stats['parse_read_errors']} \\\\
"""
    if axelrod_check_performed:
        latex_string += f"Python files not in Axelrod's list & {len(stats['files_not_in_axelrod'])} \\\\ \n"
    latex_string += """\\bottomrule
\\end{tabular}
}
"""
    if axelrod_check_performed:
        latex_string += f"""\\footnotesize{{{footnote_symbol_loc} LOC excludes comments/blank lines, calculated for {len(stats['loc_counts_matched'])} files that matched an Axelrod strategy name and could be read.}} \\\\
\\footnotesize{{{footnote_symbol_issues} Non-overridden files that matched an Axelrod strategy name but had parsing errors or missing/invalid `classifier` info (see console output).}}"""
    else:
        latex_string += f"""\\footnotesize{{{footnote_symbol_loc} LOC excludes comments/blank lines, calculated for {len(stats['loc_counts_matched'])} scanned files where LOC could be determined.}} \\\\
\\footnotesize{{{footnote_symbol_issues} Non-overridden files with parsing errors or missing/invalid `classifier` info (see console output).}}"""

    latex_string += f"""
\\end{{table}}
\\end{{document}}
"""
    try:
        with open(latex_filepath, 'w') as f:
            f.write(latex_string)
        print(f"Strategy statistics LaTeX table saved to: {latex_filepath}")
    except Exception as e:
        print(f"Error saving LaTeX table: {e}")

    print("--- Strategy Statistics Generation Complete ---")


# --- Visualization Functions ---

def create_summary_table(df, output_dir, model_configs=None, prompt_order=None, model_display_names=None):
    """
    Creates a publication-quality summary table (LaTeX and optionally PNG)
    showing accuracy for each model vs. (Perturbation Type, Prompt Strategy).
    """
    if df.empty: print("Summary Table: No data provided."); return
    if model_configs is None: model_configs = DEFAULT_MODEL_CONFIGS # Fallback
    if prompt_order is None: prompt_order = ['ZS', 'FS', 'COT'] # Default if not specified
    if model_display_names is None: model_display_names = {cfg['name']: cfg.get('display_name', cfg['name']) for cfg in model_configs}


    # Ensure 'Program_Type' and 'Prompt Strategy' are categorical with the desired order
    perturbation_order = [pt for pt in ['unmasked', 'masked', 'obfuscated'] if pt in df['Program_Type'].unique()]
    current_prompt_order = [p for p in prompt_order if p in df['Prompt Strategy'].unique()]


    df['Program_Type'] = pd.Categorical(df['Program_Type'], categories=perturbation_order, ordered=True)
    df['Prompt Strategy'] = pd.Categorical(df['Prompt Strategy'], categories=current_prompt_order, ordered=True)


    # Get the desired model order based on their source_type
    source_type_order = ['reasoning', 'closed', 'open']
    model_order_sorted = []
    for source_type in source_type_order:
        for config in model_configs:
            if config['source_type'] == source_type and config['name'] in df['Model'].unique():
                model_order_sorted.append(config['name'])
    # Add any remaining models not in the predefined source types (shouldn't happen if configs are complete)
    for model_name in df['Model'].unique():
        if model_name not in model_order_sorted:
            model_order_sorted.append(model_name)


    current_model_display_names = {name: model_display_names.get(name, name) for name in model_order_sorted}


    try:
        df['Accuracy'] = df['Correct Prediction'].astype(float) * 100
        grouped = df.groupby(['Model', 'Program_Type', 'Prompt Strategy'], observed=False)['Accuracy'].mean().reset_index()
    except Exception as e:
        print(f"Error during accuracy calculation for table: {e}"); traceback.print_exc(); return

    try:
        summary_pivot = grouped.pivot_table(index='Model', columns=['Program_Type', 'Prompt Strategy'], values='Accuracy')
    except Exception as e:
        print(f"Error pivoting data for table: {e}"); print("Grouped data for table:", grouped); return
    
    summary_pivot = summary_pivot.reindex(model_order_sorted)
    summary_pivot.index = [current_model_display_names.get(model, model) for model in summary_pivot.index]

    # Create the desired column order
    new_col_multi_index = pd.MultiIndex.from_product(
        [perturbation_order, current_prompt_order],
        names=['Perturbation', 'Prompt Strategy']
    )
    
    valid_cols = new_col_multi_index.intersection(summary_pivot.columns)
    if valid_cols.empty:
        print("Warning: No valid columns after reindexing for table."); summary_pivot_reordered = summary_pivot
    else:
        summary_pivot_reordered = summary_pivot.reindex(columns=valid_cols)
    
    summary_formatted = summary_pivot_reordered.map(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")

    latex_path = os.path.join(output_dir, "model_prompt_accuracy_summary_wide.tex")
    try:
        # Convert to LaTeX
        styler = summary_formatted.style
        styler.format(na_rep="-")
        styler.set_caption("LLM Prediction Accuracy (%) by Program Perturbation and Prompt Strategy")
        styler.set_table_styles([
            {'selector': 'toprule', 'props': ':hline;'},
            {'selector': 'midrule', 'props': ':hline;'},
            {'selector': 'bottomrule', 'props': ':hline;'},
            {'selector': 'th.col_heading', 'props': 'text-align: center;'},
            {'selector': 'td', 'props': 'text-align: right; padding: 0.3em;'}
        ])
        
        latex_string = styler.to_latex(
            hrules=True,
            multicol_align="c", # Center align multicolumn headers
            clines="all;data", # Add clines for better separation
        )
        
        latex_string = re.sub(r"Perturbation & & Prompt Strategy", r"Perturbation & \multicolumn{2}{c}{Prompt Strategy}", latex_string, count=1)

        for pt in perturbation_order:
             pass


        new_lines = []
        last_source_type_for_table = None
        model_source_map_for_table = {config["name"]: config["source_type"] for config in model_configs}

        for line_idx, line in enumerate(latex_string.splitlines()):
            new_lines.append(line)
            if line_idx > 0 and "&" in line: # Check if it's a data row
                model_display_name_in_row = line.split('&')[0].strip()
                
                original_model_name_for_row = None
                for m_name, d_name in current_model_display_names.items():
                    if d_name == model_display_name_in_row:
                        original_model_name_for_row = m_name
                        break
                
                if original_model_name_for_row:
                    current_source_type = model_source_map_for_table.get(original_model_name_for_row)
                    if last_source_type_for_table is not None and current_source_type != last_source_type_for_table:
                        if not (new_lines[-2].strip() == "\\midrule"):
                             new_lines.insert(-1, "\\midrule") 
                    last_source_type_for_table = current_source_type
        
        latex_string_final = '\n'.join(new_lines)

        if latex_string_final:
            with open(latex_path, 'w') as f:
                f.write(latex_string_final)
            print(f"Widely grouped LaTeX summary table saved to: {latex_path}")
        else:
            print("Error: Final LaTeX string was empty.")

    except Exception as e:
        print(f"Error generating widely grouped LaTeX table: {e}"); traceback.print_exc()


    if HAS_DFI:
        png_path = os.path.join(output_dir, "model_prompt_accuracy_summary_wide.png")
        try:
            df_for_png = pd.DataFrame() 
            
            last_source_type_png = None
            temp_index_col_name = "___Model_Index___" 
            summary_formatted_with_index = summary_formatted.reset_index()
            
            original_index_name_in_df = summary_formatted.index.name if summary_formatted.index.name is not None else 'index'
            if original_index_name_in_df not in summary_formatted_with_index.columns and 'index' in summary_formatted_with_index.columns:
                original_index_name_in_df = 'index' 
            
            if original_index_name_in_df not in summary_formatted_with_index.columns:
                print(f"Warning for PNG: Original index name '{original_index_name_in_df}' not found in columns after reset_index. Columns are: {summary_formatted_with_index.columns}")
                styled_df = summary_formatted_with_index.style # Apply style directly
            else:
                summary_formatted_with_index = summary_formatted_with_index.rename(columns={original_index_name_in_df: temp_index_col_name})
                styled_df = summary_formatted_with_index.style

            styled_df = styled_df.set_caption("LLM Prediction Accuracy (%)") \
                .set_table_styles([
                    {'selector': 'th', 'props': [('background-color', '#f2f2f2'), ('text-align', 'center')]},
                    {'selector': 'td', 'props': [('text-align', 'right'), ('padding', '0.3em')]},
                    {'selector': 'caption', 'props': [('caption-side', 'bottom'), ('font-size', '0.9em')]}
                ])
            
            if hasattr(styled_df, 'hide_index_'): # Check for older pandas versions
                 styled_df = styled_df.hide_index_() 
            elif hasattr(styled_df, 'hide'): # For newer pandas versions
                 styled_df = styled_df.hide(axis='index') # Hides the DataFrame index
            else:
                print("Warning for PNG: Neither hide_index_ nor hide method found on Styler object.")


            styled_df = styled_df.set_properties(**{'border': '1px solid black', 'width': '80px'})

            dfi.export(styled_df, png_path, dpi=300, table_conversion='chrome')
            print(f"Widely grouped PNG summary table saved to: {png_path}")
        except Exception as e:
            print(f"Could not save widely grouped summary table as PNG: {e}"); traceback.print_exc()
            csv_path = os.path.join(output_dir, "model_prompt_accuracy_summary_wide_fallback.csv")
            summary_formatted.to_csv(csv_path)
            print(f"Formatted wide data saved to fallback CSV: {csv_path}")
    else:
        print("Install 'dataframe_image' to save the summary table as PNG.")
        csv_path = os.path.join(output_dir, "model_prompt_accuracy_summary_wide.csv")
        summary_formatted.to_csv(csv_path)
        print(f"Formatted wide CSV summary table saved to: {csv_path}")


# --- Helper function to load data ---
def load_experiment_data(results_dir, model_configs):
    """Loads and preprocesses experiment data from CSV files."""
    all_files = glob.glob(os.path.join(results_dir, "**", "*.csv"), recursive=True)

    if not all_files:
        print(f"No CSV files found in {results_dir} or its subdirectories.")
        return pd.DataFrame()

    df_list = []
    print(f"Found {len(all_files)} CSV files to process.") 
    for f in all_files:
        try:
            print(f"Reading file: {f}") 
            df_temp = pd.read_csv(f)
            if 'Model' not in df_temp.columns:
                print(f"Warning: 'Model' column missing in {f}. Skipping this file.")
                continue
            model_to_source = {config['name']: config.get('source_type', 'unknown') for config in model_configs}
            df_temp['source_type'] = df_temp['Model'].map(model_to_source).fillna('unknown')
            df_list.append(df_temp)
        except Exception as e:
            print(f"Error reading or processing {f}: {e}")

    if not df_list:
        print("No data loaded after processing CSV files.")
        return pd.DataFrame()

    df_main = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(df_main)} rows from {len(df_list)} successfully processed CSV files.")

    if 'Correct Prediction' in df_main.columns:
        if df_main['Correct Prediction'].dtype == 'object': 
             df_main['Correct Prediction'] = df_main['Correct Prediction'].str.lower().map({'yes': True, 'true': True, 'no': False, 'false': False, 'nan': False}).fillna(False)
        df_main['Correct Prediction'] = df_main['Correct Prediction'].astype(bool)
    else:
        print("Warning: 'Correct Prediction' column not found in the loaded data.")
        df_main['Correct Prediction'] = False

    if 'Program_Type' not in df_main.columns:
        print("Warning: 'Program_Type' column not found. Defaulting to 'unmasked'.")
        df_main['Program_Type'] = 'unmasked'
    else:
        df_main['Program_Type'] = df_main['Program_Type'].fillna('unknown').str.lower()
        df_main['Program_Type'] = df_main['Program_Type'].replace(['unmasked', 'original', 'refactored'], 'unmasked')

    if 'Stochastic' in df_main.columns:
        if df_main['Stochastic'].dtype == 'object':
            df_main['Stochastic'] = df_main['Stochastic'].str.lower().map({'true': True, 'false': False}).fillna(False) 
        df_main['Stochastic'] = df_main['Stochastic'].astype(bool)
    else:
        print("Warning: 'Stochastic' column not found. Cannot perform stochastic/deterministic comparison.")

    key_cols_for_grouping = ['Model', 'Prompt Strategy', 'Program_Type', 'Stochastic']
    for col in key_cols_for_grouping:
        if col in df_main.columns:
            if df_main[col].isnull().any():
                print(f"Warning: Found NaN values in '{col}'. Filling with 'unknown' or False for boolean.")
                if df_main[col].dtype == 'bool':
                    df_main[col] = df_main[col].fillna(False)
                else:
                    df_main[col] = df_main[col].fillna('unknown')
    
    if 'Program_Type' in df_main.columns:
        print("Unique Program_Type values found:", df_main['Program_Type'].unique())
    else:
        print("Error: 'Program_Type' column is missing. Grouped table cannot be generated.")
    return df_main

# --- Other visualization functions ---

def create_confusion_matrix(df, output_file, title):
    """
    Creates and saves a confusion matrix plot.
    """
    if df.empty:
        print(f"No valid data for confusion matrix: {title}"); return
    if 'Prediction' not in df.columns or 'Ground Truth' not in df.columns:
        print(f"Warning: Prediction/Ground Truth missing for CM: {title}."); return

    df_copy = df.copy()
    valid_preds = ["yes", "no"]; 
    df_copy = df_copy[df_copy['Prediction'].isin(valid_preds) & df_copy['Ground Truth'].isin(valid_preds)].copy()
    
    if df_copy.empty:
        print(f"No valid 'yes'/'no' predictions found for CM: {title}. Skipping."); return

    categories = ['yes', 'no']
    df_copy['Prediction'] = pd.Categorical(df_copy['Prediction'], categories=categories, ordered=True)
    df_copy['Ground Truth'] = pd.Categorical(df_copy['Ground Truth'], categories=categories, ordered=True)

    cm = pd.crosstab(df_copy['Ground Truth'], df_copy['Prediction'], dropna=False)
    cm = cm.reindex(index=categories, columns=categories, fill_value=0)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual (Ground Truth)')
    plt.tight_layout()
    try:
        plt.savefig(output_file, format="pdf", bbox_inches='tight')
        print(f"Confusion matrix saved to: {output_file}")
    except Exception as e:
        print(f"Error saving confusion matrix {output_file}: {e}")
    plt.close()


def calculate_accuracy_metrics(df_group):
    """Calculates various accuracy metrics for a given DataFrame group."""
    if df_group.empty or 'Correct Prediction' not in df_group.columns:
        return {
            'total_predictions': 0, 'correct_predictions': 0, 'accuracy': 0.0,
            'true_positive': 0, 'true_negative': 0, 'false_positive': 0, 'false_negative': 0,
            'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0
        }
    
    df_group['Correct Prediction'] = df_group['Correct Prediction'].astype(bool)
    correct = df_group['Correct Prediction'].sum()
    total = len(df_group)
    acc = (correct / total) * 100 if total > 0 else 0.0

    tp, tn, fp, fn = 0, 0, 0, 0
    if 'Prediction' in df_group.columns and 'Ground Truth' in df_group.columns:
        pred_bool = df_group['Prediction'].astype(str).str.lower().map({'yes': True, 'true': True, 'no': False, 'false': False})
        truth_bool = df_group['Ground Truth'].astype(str).str.lower().map({'yes': True, 'true': True, 'no': False, 'false': False})
        tp = ((pred_bool == True) & (truth_bool == True)).sum()
        tn = ((pred_bool == False) & (truth_bool == False)).sum()
        fp = ((pred_bool == True) & (truth_bool == False)).sum()
        fn = ((pred_bool == False) & (truth_bool == True)).sum()

    prec = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
    rec = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    
    return {
        'total_predictions': total, 'correct_predictions': correct, 'accuracy': acc,
        'true_positive': tp, 'true_negative': tn, 'false_positive': fp, 'false_negative': fn,
        'precision': prec, 'recall': rec, 'f1_score': f1
    }


def create_accuracy_plot_by_model(df, output_path, title, program_type_label="Program", strategy_type_label="Full", model_configs=None, model_display_names=None, prompt_order=None):
    if df.empty:
        print(f"Plotting: No data for {title if title else 'Accuracy Plot'} ({program_type_label} - {strategy_type_label}). Skipping."); return
    if model_configs is None: model_configs = DEFAULT_MODEL_CONFIGS 
    if prompt_order is None: prompt_order = ['ZS', 'FS', 'COT'] 
    
    if model_display_names is None:
        current_model_display_names = {config["name"]: config.get("display_name", config["name"]) for config in model_configs}
    else:
        current_model_display_names = model_display_names

    models_in_data = df['Model'].unique()
    source_type_order = ['reasoning', 'closed', 'open'] 
    plot_model_order = []
    for source_type in source_type_order:
        for config in model_configs:
            if config['source_type'] == source_type and config['name'] in models_in_data:
                plot_model_order.append(current_model_display_names.get(config['name'], config['name']))
    
    display_names_in_data = [current_model_display_names.get(m, m) for m in models_in_data]
    for model_disp_name in display_names_in_data:
        if model_disp_name not in plot_model_order:
            plot_model_order.append(model_disp_name)

    plot_prompt_order = [p for p in prompt_order if p in df['Prompt Strategy'].unique()]
    
    try:
        plot_df = df.groupby(['Model', 'Prompt Strategy'], observed=False).apply(
            lambda x: calculate_accuracy_metrics(x)['accuracy'], include_groups=False
        ).reset_index(name='Accuracy')
    except Exception as e:
        print(f"Error grouping/calculating accuracy for plot {title if title else 'Accuracy Plot'}: {e}")
        traceback.print_exc(); return

    if plot_df.empty or 'Accuracy' not in plot_df.columns or plot_df['Accuracy'].isnull().all():
        print(f"Plotting: No accuracy data to plot for {title if title else 'Accuracy Plot'}. Skipping."); return

    plot_df['Model_Display'] = plot_df['Model'].map(current_model_display_names)
    plot_df = plot_df.dropna(subset=['Model_Display'])
    if plot_df.empty :
        print(f"Plotting: No models left after mapping display names for {title if title else 'Accuracy Plot'}. Skipping."); return

    sns.set_context("talk", font_scale=2.0) 
    plt.figure(figsize=(max(12, len(plot_model_order) * 1.5), 7)) 
    palette = {'ZS': '#2878B5', 'FS': '#FA7F6F', 'COT': '#8EBA42'}
    for p_strat in plot_prompt_order:
        if p_strat not in palette:
            palette[p_strat] = sns.color_palette("bright", len(plot_prompt_order))[plot_prompt_order.index(p_strat) % len(sns.color_palette("bright"))]

    ax = sns.barplot(x='Model_Display', y='Accuracy', hue='Prompt Strategy', data=plot_df, 
                     palette=palette, order=plot_model_order, hue_order=plot_prompt_order,
                     edgecolor='black', linewidth=0.8) 
    
    plt.title(title, fontsize=32)
    plt.ylabel('Prediction Accuracy (%)', fontsize=32)
    plt.xlabel('Model', fontsize=32)
    plt.xticks(rotation=45, ha='right', fontsize=32)
    plt.yticks(fontsize=32)
    plt.ylim(0, 100)
    
    for p_patch in ax.patches: 
        height = p_patch.get_height()
        if pd.notna(height) and height > 0: 
            ax.annotate(f'{height:.1f}%', 
                        (p_patch.get_x() + p_patch.get_width() / 2., height),
                        ha='center', va='bottom', xytext=(0, 4), 
                        textcoords='offset points', fontsize=8, color='black')

    ax.legend(title='Prompt Strategy', fontsize=32, title_fontsize=12, loc='upper right', frameon=True, shadow=False)
    ax.yaxis.grid(False) # Remove y-axis grid lines specifically
    sns.despine() 
    plt.tight_layout()
    
    try:
        plt.savefig(output_path, format="pdf", bbox_inches='tight')
        print(f"Accuracy plot saved to: {output_path}")
    except Exception as e:
        print(f"Error saving plot {output_path}: {e}")
    plt.close()

def create_summary_statistics(df_main, output_dir):
    if df_main.empty: print("Summary Stats: No data provided."); return
    summary_data = []
    grouped_overall = df_main.groupby(['Model', 'Program_Type', 'Prompt Strategy'], observed=True)

    for name, group in grouped_overall:
        metrics = calculate_accuracy_metrics(group)
        summary_data.append({
            'Model': name[0], 'Program_Type': name[1], 'Prompt Strategy': name[2],
            'Total Predictions': metrics['total_predictions'], 'Correct Predictions': metrics['correct_predictions'],
            'Accuracy (%)': metrics['accuracy'], 'Precision (%)': metrics['precision'],
            'Recall (%)': metrics['recall'], 'F1 Score (%)': metrics['f1_score']
        })
    
    if not summary_data: print("No data to summarize after grouping."); return
    summary_df = pd.DataFrame(summary_data)
    cols_order = ['Model', 'Program_Type', 'Prompt Strategy', 
                  'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1 Score (%)', 
                  'Correct Predictions', 'Total Predictions']
    summary_df = summary_df[[col for col in cols_order if col in summary_df.columns]]
    summary_df.rename(columns={'accuracy': 'Accuracy (%)', 'precision': 'Precision (%)', 
                               'recall': 'Recall (%)', 'f1_score': 'F1 Score (%)'}, inplace=True)
    for col in ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1 Score (%)']:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].round(2)

    csv_path = os.path.join(output_dir, "summary_statistics.csv"); summary_df.to_csv(csv_path, index=False)
    print(f"Summary statistics saved to: {csv_path}")
    excel_path = os.path.join(output_dir, "summary_statistics.xlsx")
    try:
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            workbook = writer.book; worksheet = writer.sheets['Summary']
            header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#D7E4BC', 'border': 1})
            for col_num, value in enumerate(summary_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                column_len = max(summary_df[value].astype(str).map(len).max(), len(value))
                worksheet.set_column(col_num, col_num, column_len + 2) 
        print(f"Summary statistics Excel saved to: {excel_path}")
    except Exception as e:
        print(f"Error saving summary statistics to Excel: {e}. Ensure 'xlsxwriter' is installed.")

def perform_and_write_ttest(group1_accuracies, group2_accuracies, group1_name, group2_name, comparison_title, outfile):
    if len(group1_accuracies) < 2 or len(group2_accuracies) < 2:
        outfile.write(f"  Skipping t-test for {group1_name} vs {group2_name} (insufficient data).\n")
        print(f"Skipping t-test for {group1_name} vs {group2_name} (insufficient data: {len(group1_accuracies)} vs {len(group2_accuracies)}).")
        return
    t_stat, p_value = ttest_ind(group1_accuracies, group2_accuracies, equal_var=False, nan_policy='omit')
    outfile.write(f"  Comparison: {group1_name} vs {group2_name}\n")
    outfile.write(f"    T-statistic: {t_stat:.4f}\n    P-value: {p_value:.4f}\n")
    outfile.write(f"    Result: Statistically {'significant (p < 0.05)' if p_value < 0.05 else 'not significant (p >= 0.05)'}\n\n")
    print(f"  T-test for {group1_name} vs {group2_name}: t={t_stat:.4f}, p={p_value:.4f}")

def plot_accuracy_comparison_zs_vs_cot(df_main, output_dir, model_configs=None, model_display_names=None, t_test_file=None):
    if df_main.empty: print("ZS vs COT: No data provided."); return
    if model_configs is None: model_configs = DEFAULT_MODEL_CONFIGS 
    if model_display_names is None: current_model_display_names = {config["name"]: config.get("display_name", config["name"]) for config in model_configs}
    else: current_model_display_names = model_display_names

    df_filtered = df_main[df_main['Prompt Strategy'].isin(['ZS', 'COT']) & (df_main['Program_Type'] == 'unmasked')].copy()
    if df_filtered.empty: print("ZS vs COT: No data after filtering for ZS/COT unmasked."); return
    df_filtered['Accuracy'] = df_filtered['Correct Prediction'].astype(float) * 100
    model_accuracies = df_filtered.groupby(['Model', 'Prompt Strategy'], observed=False)['Accuracy'].mean().unstack()
    model_accuracies = model_accuracies.dropna(subset=['ZS', 'COT'], how='any')
    if model_accuracies.empty: print("ZS vs COT: No models with both ZS and COT unmasked runs after filtering."); return

    avg_zs_acc = model_accuracies['ZS'].mean(skipna=True) 
    avg_cot_acc = model_accuracies['COT'].mean(skipna=True)
    sem_zs_acc = model_accuracies['ZS'].sem(skipna=True) if model_accuracies['ZS'].notna().sum() > 1 else 0
    sem_cot_acc = model_accuracies['COT'].sem(skipna=True) if model_accuracies['COT'].notna().sum() > 1 else 0
    
    plot_data = model_accuracies.copy()
    plot_data.index = plot_data.index.map(current_model_display_names) 
    if plot_data.isnull().all().all(): print("ZS vs COT: All accuracy data is NaN after mapping display names."); return
        
    sns.set_context("talk", font_scale=2.5) 
    fig, ax = plt.subplots(figsize=(12, 7)) 
    bar_positions = [0, 1]; bar_labels = ['Zero-Shot (ZS)', 'Chain of Thought (COT)']
    avg_accuracies = [avg_zs_acc, avg_cot_acc]; error_values = [sem_zs_acc, sem_cot_acc]
    ax.bar(bar_positions, avg_accuracies, yerr=error_values, capsize=5, width=0.35, color=['#FFFFFF', '#FFFFFF'], alpha=0.7, edgecolor='black', zorder=2)

    individual_model_color = 'lightgrey'; line_dot_alpha = 0.7; dot_size = 8 
    for i, model_row in plot_data.iterrows():
        zs_acc, cot_acc = model_row['ZS'], model_row['COT']
        ax.plot(bar_positions[0], zs_acc, color=individual_model_color, marker='o', markersize=dot_size, alpha=line_dot_alpha, zorder=3)
        ax.plot(bar_positions[1], cot_acc, color=individual_model_color, marker='o', markersize=dot_size, alpha=line_dot_alpha, zorder=3)
        ax.plot(bar_positions, [zs_acc, cot_acc], color=individual_model_color, linestyle='-', alpha=line_dot_alpha, zorder=1, linewidth=1.5)

    ax.set_title("Zero-Shot vs Chain of Thought", fontsize=32) # Added title
    ax.set_xticks(bar_positions); ax.set_xticklabels(bar_labels, fontsize=32)
    ax.set_ylabel('Prediction Accuracy (%)', fontsize=32)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(False) 
    sns.despine() 
    plt.tight_layout(rect=[0, 0, 1, 1]) 
    
    output_file_plot = os.path.join(output_dir, "zs_vs_cot_comparison.pdf")
    try:
        plt.savefig(output_file_plot, format="pdf", bbox_inches='tight')
        print(f"ZS vs COT comparison plot saved to: {output_file_plot}")
    except Exception as e: print(f"Error saving ZS vs COT plot: {e}")
    plt.close()

    if t_test_file:
        zs_accuracies_for_ttest = model_accuracies['ZS'].dropna() 
        cot_accuracies_for_ttest = model_accuracies['COT'].dropna() 
        t_test_file.write("Comparison: Zero-Shot (ZS) vs Chain of Thought (COT) (Unmasked Strategies - Filtered for models with both ZS & COT data)\n")
        perform_and_write_ttest(zs_accuracies_for_ttest, cot_accuracies_for_ttest, "ZS Accuracy (Filtered)", "COT Accuracy (Filtered)", "ZS vs COT (Filtered)", t_test_file)

def plot_accuracy_perturbation_comparison_zs(df_main, output_dir, model_configs=None, model_display_names=None, t_test_file=None):
    if df_main.empty: print("Perturbation Comp: No data provided."); return
    if model_configs is None: model_configs = DEFAULT_MODEL_CONFIGS
    if model_display_names is None: current_model_display_names = {config["name"]: config.get("display_name", config["name"]) for config in model_configs}
    else: current_model_display_names = model_display_names

    df_filtered = df_main[df_main['Prompt Strategy'] == 'ZS'].copy()
    if df_filtered.empty: print("Perturbation Comp: No ZS data found."); return
    df_filtered['Accuracy'] = df_filtered['Correct Prediction'].astype(float) * 100
    perturb_types = ['unmasked', 'masked', 'obfuscated'] 
    df_filtered['Program_Type'] = pd.Categorical(df_filtered['Program_Type'], categories=perturb_types, ordered=True)
    model_accuracies_all_perturbs = df_filtered.groupby(['Model', 'Program_Type'], observed=False)['Accuracy'].mean().unstack()
    model_accuracies_all_perturbs = model_accuracies_all_perturbs.reindex(columns=perturb_types).dropna(how='all')
    if model_accuracies_all_perturbs.empty: print("Perturbation Comp: No models with ZS runs for any perturbation type after grouping."); return

    avg_accuracies_perturbs = [model_accuracies_all_perturbs[pt].mean(skipna=True) for pt in perturb_types]
    sem_errors_perturbs = [model_accuracies_all_perturbs[pt].sem(skipna=True) if model_accuracies_all_perturbs[pt].notna().sum() > 1 else 0 for pt in perturb_types]
    plot_data = model_accuracies_all_perturbs.reset_index()
    plot_data['Model_Display'] = plot_data['Model'].map(current_model_display_names)
    plot_data = plot_data.dropna(subset=['Model_Display']) 
    if plot_data.empty: print("Perturbation Comp: No models left after mapping display names."); return
    
    sns.set_context("talk", font_scale=2.5) 
    fig, ax = plt.subplots(figsize=(12, 7)) 
    bar_positions = np.arange(len(perturb_types)); bar_labels = [pt.capitalize() for pt in perturb_types]
    bar_colors_map = ['#1f77b4', '#ff7f0e', '#2ca02c'] 
    ax.bar(bar_positions, avg_accuracies_perturbs, yerr=sem_errors_perturbs, capsize=5, color='white', alpha=0.7, edgecolor='black', width=0.45, zorder=2)

    individual_model_color = 'lightgrey'; line_dot_alpha = 0.7; dot_size = 8 
    for i, model_row in plot_data.iterrows():
        accuracies_for_model = [model_row[pt] for pt in perturb_types]
        valid_points_x, valid_points_y = [], []
        for idx, acc_val in enumerate(accuracies_for_model):
            if pd.notna(acc_val):
                ax.plot(bar_positions[idx], acc_val, color=individual_model_color, marker='o', markersize=dot_size, alpha=line_dot_alpha, zorder=3)
                valid_points_x.append(bar_positions[idx]); valid_points_y.append(acc_val)
        if len(valid_points_x) > 1:
             ax.plot(valid_points_x, valid_points_y, color=individual_model_color, linestyle='-', alpha=line_dot_alpha, zorder=1, linewidth=1.5)

    ax.set_title("Perturbation Type", fontsize=32)
    ax.set_xticks(bar_positions); ax.set_xticklabels(bar_labels, fontsize=32)
    ax.set_ylabel('Prediction Accuracy (%)', fontsize=32)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(False) # FIX: Correct way to turn off y-axis grid
    sns.despine()
    plt.tight_layout()
    
    output_file_plot = os.path.join(output_dir, "perturbation_comparison_zs.pdf")
    try:
        plt.savefig(output_file_plot, format="pdf", bbox_inches='tight')
        print(f"Perturbation comparison plot saved to: {output_file_plot}")
    except Exception as e: print(f"Error saving perturbation plot: {e}")
    plt.close()

    if t_test_file:
        t_test_file.write("Comparison: Perturbation Types (ZS Prompt Strategy)\n")
        unmasked_accuracies = df_filtered[df_filtered['Program_Type'] == 'unmasked']['Accuracy'].dropna()
        masked_accuracies = df_filtered[df_filtered['Program_Type'] == 'masked']['Accuracy'].dropna()
        obfuscated_accuracies = df_filtered[df_filtered['Program_Type'] == 'obfuscated']['Accuracy'].dropna()
        perform_and_write_ttest(unmasked_accuracies, masked_accuracies, "Unmasked", "Masked", "Perturbation ZS", t_test_file)
        perform_and_write_ttest(unmasked_accuracies, obfuscated_accuracies, "Unmasked", "Obfuscated", "Perturbation ZS", t_test_file)
        perform_and_write_ttest(masked_accuracies, obfuscated_accuracies, "Masked", "Obfuscated", "Perturbation ZS", t_test_file)

def plot_stochastic_vs_deterministic_zs_unmasked(df_main, output_dir, model_configs=None, model_display_names=None, t_test_file=None):
    if df_main.empty: print("Stochastic/Det: No data provided."); return
    if 'Stochastic' not in df_main.columns: print("Stochastic/Det: 'Stochastic' column missing."); return
    if model_configs is None: model_configs = DEFAULT_MODEL_CONFIGS
    if model_display_names is None: current_model_display_names = {config["name"]: config.get("display_name", config["name"]) for config in model_configs}
    else: current_model_display_names = model_display_names

    df_filtered = df_main[(df_main['Prompt Strategy'] == 'ZS') & (df_main['Program_Type'] == 'unmasked')].copy()
    if df_filtered.empty: print("Stochastic/Det: No ZS unmasked data found."); return
    if 'Stochastic' not in df_filtered.columns: print("Stochastic/Det: 'Stochastic' column missing from filtered ZS unmasked data."); return

    df_filtered['Accuracy'] = df_filtered['Correct Prediction'].astype(float) * 100
    df_filtered['Strategy_Kind'] = df_filtered['Stochastic'].map({True: 'Stochastic', False: 'Deterministic'})
    model_accuracies = df_filtered.groupby(['Model', 'Strategy_Kind'], observed=False)['Accuracy'].mean().unstack()
    if 'Deterministic' not in model_accuracies.columns: model_accuracies['Deterministic'] = np.nan
    if 'Stochastic' not in model_accuracies.columns: model_accuracies['Stochastic'] = np.nan
    model_accuracies_filtered = model_accuracies[['Deterministic', 'Stochastic']].dropna(how='all')
    if model_accuracies_filtered.empty: print("Stochastic/Det: No models found with ZS unmasked runs for EITHER Stochastic OR Deterministic strategies."); return

    avg_det_acc = model_accuracies_filtered['Deterministic'].mean(skipna=True)
    avg_sto_acc = model_accuracies_filtered['Stochastic'].mean(skipna=True)
    sem_det_acc = model_accuracies_filtered['Deterministic'].sem(skipna=True) if model_accuracies_filtered['Deterministic'].notna().sum() > 1 else 0
    sem_sto_acc = model_accuracies_filtered['Stochastic'].sem(skipna=True) if model_accuracies_filtered['Stochastic'].notna().sum() > 1 else 0
    plot_data = model_accuracies_filtered.copy()
    plot_data.index = plot_data.index.map(current_model_display_names)
    if plot_data.isnull().all().all(): print("Stochastic/Det: All accuracy data is NaN after mapping display names."); return

    sns.set_context("talk", font_scale=2.5) 
    fig, ax = plt.subplots(figsize=(12, 7)) 
    bar_positions = [0, 1]; bar_labels = ['Deterministic', 'Stochastic']
    avg_accuracies = [avg_det_acc, avg_sto_acc]; error_values = [sem_det_acc, sem_sto_acc]
    ax.bar(bar_positions, avg_accuracies, yerr=error_values, capsize=5, width=0.35, color=['#FFFFFF', '#FFFFFF'], alpha=0.7, edgecolor='black', zorder=2) 

    individual_model_color = 'lightgrey'; line_dot_alpha = 0.7; dot_size = 8 
    for i, model_row in plot_data.iterrows():
        det_acc, sto_acc = model_row['Deterministic'], model_row['Stochastic']
        if pd.notna(det_acc): ax.plot(bar_positions[0], det_acc, color=individual_model_color, marker='o', markersize=dot_size, alpha=line_dot_alpha, zorder=3)
        if pd.notna(sto_acc): ax.plot(bar_positions[1], sto_acc, color=individual_model_color, marker='o', markersize=dot_size, alpha=line_dot_alpha, zorder=3)
        if pd.notna(det_acc) and pd.notna(sto_acc):
             ax.plot(bar_positions, [det_acc, sto_acc], color=individual_model_color, linestyle='-', alpha=line_dot_alpha, zorder=1, linewidth=1.5)
    
    ax.set_title("Deterministic vs Stochastic Strategies", fontsize=32) 
    ax.set_xticks(bar_positions); ax.set_xticklabels(bar_labels, fontsize=32)
    ax.set_ylabel('Prediction Accuracy (%)', fontsize=32)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(False) 
    sns.despine()
    plt.tight_layout() 
    
    output_file_plot = os.path.join(output_dir, "stochastic_vs_deterministic_zs_unmasked.pdf")
    try:
        plt.savefig(output_file_plot, format="pdf", bbox_inches='tight')
        print(f"Stochastic vs Deterministic plot saved to: {output_file_plot}")
    except Exception as e: print(f"Error saving Stochastic vs Deterministic plot: {e}")
    plt.close()
    
    if t_test_file:
        deterministic_accuracies = df_filtered[df_filtered['Stochastic'] == False]['Accuracy'].dropna()
        stochastic_accuracies = df_filtered[df_filtered['Stochastic'] == True]['Accuracy'].dropna()
        t_test_file.write("Comparison: Deterministic vs Stochastic Strategies (ZS Unmasked)\n")
        perform_and_write_ttest(deterministic_accuracies, stochastic_accuracies, "Deterministic Accuracy", "Stochastic Accuracy", "Deterministic vs Stochastic", t_test_file)

# --- Main Visualization Function ---
def visualize_all_results(results_dir, output_dir, model_configs=None, model_display_names=None, prompt_order=None):
    print(f"\n--- Starting Visualization Process ---")
    print(f"Reading results from: {results_dir}")
    print(f"Saving visualizations to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    t_test_filepath = os.path.join(output_dir, "t-test.txt")
    with open(t_test_filepath, 'w') as t_test_file: 
        t_test_file.write(f"T-Test Results - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        df_main = load_experiment_data(results_dir, model_configs if model_configs else DEFAULT_MODEL_CONFIGS)
        if df_main.empty:
            print("No data loaded. Exiting visualization."); t_test_file.write("No data loaded to perform t-tests.\n"); return

        current_model_configs = model_configs if model_configs else DEFAULT_MODEL_CONFIGS
        current_model_display_names = model_display_names if model_display_names else {config["name"]: config.get("display_name", config["name"]) for config in current_model_configs}
        current_prompt_order = prompt_order if prompt_order else ['ZS', 'FS', 'COT']

        program_type_col, program_type_map, program_types_found = None, {}, []
        if 'Program_Type' in df_main.columns:
            program_type_col = 'Program_Type'; available_values = df_main[program_type_col].unique()
            if 'unmasked' in available_values: program_type_map["Unmasked"] = "unmasked"; program_types_found.append("Unmasked")
            if 'masked' in available_values: program_type_map["Masked"] = "masked"; program_types_found.append("Masked")
            if 'obfuscated' in available_values: program_type_map["Obfuscated"] = "obfuscated"; program_types_found.append("Obfuscated")
        elif 'Masked' in df_main.columns: 
            program_type_col = 'Masked'; program_type_map = {"Unmasked": False, "Masked": True}
            if False in df_main[program_type_col].unique(): program_types_found.append("Unmasked")
            if True in df_main[program_type_col].unique(): program_types_found.append("Masked")
        if not program_types_found: program_types_found.append("Unmasked")

        stats_table_dir = os.path.join(output_dir, "summary_tables"); os.makedirs(stats_table_dir, exist_ok=True)
        create_summary_statistics(df_main, stats_table_dir)
        create_summary_table(df_main, stats_table_dir, model_configs=current_model_configs, prompt_order=current_prompt_order, model_display_names=current_model_display_names)
        unmasked_strategies_dir = "./unmasked" 
        if os.path.isdir(unmasked_strategies_dir): generate_strategy_stats_table(unmasked_strategies_dir, stats_table_dir)
        else: print(f"Warning: Directory '{unmasked_strategies_dir}' not found. Skipping detailed strategy stats table for unmasked.")
        
        print("\nGenerating Confusion Matrices..."); cm_dir = os.path.join(output_dir, "confusion_matrices"); os.makedirs(cm_dir, exist_ok=True)
        for model_name, group_model in df_main.groupby('Model', observed=True):
            for (prog_type, prompt_strat), group_prog_prompt in group_model.groupby(['Program_Type', 'Prompt Strategy'], observed=True):
                cm_title = f"CM: {current_model_display_names.get(model_name, model_name)} - {prog_type} - {prompt_strat}"
                output_file_cm = os.path.join(cm_dir, f"cm_{model_name.replace('/', '_')}_{prog_type}_{prompt_strat}.pdf")
                try: create_confusion_matrix(group_prog_prompt, output_file_cm, cm_title)
                except Exception as e_cm_inner: print(f"ERROR creating CM for {output_file_cm}: {e_cm_inner}")
        
        print("\nGenerating Accuracy Plots (by model, prompt strategy)...")
        plot_configs = [{"func": create_accuracy_plot_by_model, "dir_name": "accuracy_plots_by_model", "prefix": "accuracy_model_prompt", "title": ""}]
        for config_item in plot_configs:
            plot_func_item, plot_dir_item = config_item["func"], os.path.join(output_dir, config_item["dir_name"])
            os.makedirs(plot_dir_item, exist_ok=True); print(f"--- Generating plots in: {plot_dir_item} ---")
            for pt_label, pt_val in program_type_map.items():
                df_subset_prog = df_main[df_main[program_type_col] == pt_val] if program_type_col else df_main.copy()
                if df_subset_prog.empty: continue
                plot_file = os.path.join(plot_dir_item, f"{config_item['prefix']}_{pt_val}_all_strategies.pdf")
                plot_func_item(df_subset_prog, plot_file, title="", program_type_label=pt_label, strategy_type_label="All", model_configs=current_model_configs, model_display_names=current_model_display_names, prompt_order=current_prompt_order)
                if 'Stochastic' in df_subset_prog.columns:
                    df_stochastic = df_subset_prog[df_subset_prog['Stochastic'] == True]
                    plot_file_sto = os.path.join(plot_dir_item, f"{config_item['prefix']}_{pt_val}_stochastic.pdf")
                    plot_func_item(df_stochastic, plot_file_sto, title="", program_type_label=pt_label, strategy_type_label="Stochastic", model_configs=current_model_configs, model_display_names=current_model_display_names, prompt_order=current_prompt_order)
                    df_deterministic = df_subset_prog[df_subset_prog['Stochastic'] == False]
                    plot_file_det = os.path.join(plot_dir_item, f"{config_item['prefix']}_{pt_val}_deterministic.pdf")
                    plot_func_item(df_deterministic, plot_file_det, title="", program_type_label=pt_label, strategy_type_label="Deterministic", model_configs=current_model_configs, model_display_names=current_model_display_names, prompt_order=current_prompt_order)
                else: print(f"Skipping stochastic/deterministic breakdown for {pt_label} as 'Stochastic' column is missing.")

        print("\nGenerating Comparative Bar Graphs..."); comparative_plot_dir = os.path.join(output_dir, "comparative_plots"); os.makedirs(comparative_plot_dir, exist_ok=True)
        plot_accuracy_comparison_zs_vs_cot(df_main, comparative_plot_dir, model_configs=current_model_configs, model_display_names=current_model_display_names, t_test_file=t_test_file)
        plot_accuracy_perturbation_comparison_zs(df_main, comparative_plot_dir, model_configs=current_model_configs, model_display_names=current_model_display_names, t_test_file=t_test_file)
        if 'Stochastic' in df_main.columns:
            plot_stochastic_vs_deterministic_zs_unmasked(df_main, comparative_plot_dir, model_configs=current_model_configs, model_display_names=current_model_display_names, t_test_file=t_test_file)
        else:
            print("Skipping Stochastic vs Deterministic plot as 'Stochastic' column is missing from main DataFrame.")
            t_test_file.write("Comparison: Deterministic vs Stochastic Strategies (ZS Unmasked)\n  Skipping t-test (Stochastic column missing from data).\n\n")
    print(f"\n--- Visualization process completed. Check results in: {output_dir} ---")

# --- Main execution block ---
if __name__ == "__main__":
    import argparse
    import sys 
    parser = argparse.ArgumentParser(description="Visualize IPD experiment results")
    parser.add_argument("--results_dir", default="results", help="Directory containing experiment results (CSV files)")
    parser.add_argument("--output_dir", default="visualizations", help="Directory to save visualization outputs")
    cli_args = parser.parse_args([] if any(arg.startswith('-f') or 'ipykernel_launcher' in arg for arg in sys.argv) else None) 
    if 'ipykernel_launcher' in sys.argv[0] or (cli_args is not None and not vars(cli_args)): # Basic check for Jupyter/similar
        print("IPython/Jupyter environment detected or no CLI args. Using default arguments for argparse.")
        cli_args = parser.parse_args([]) # Use default args

    main_model_configs = DEFAULT_MODEL_CONFIGS 
    main_model_display_names = {config["name"]: config.get("display_name", config["name"]) for config in main_model_configs}
    print("Running visualization script standalone...")
    visualize_all_results(cli_args.results_dir, cli_args.output_dir, model_configs=main_model_configs, model_display_names=main_model_display_names)
