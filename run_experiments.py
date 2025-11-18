"""
Main script to run IPD experiments and generate visualizations.
"""

#%%

import os
import argparse
from pathlib import Path
import axelrod as axl
from ipd_experiment import run_experiment # Assuming ipd_experiment.py is in the same directory or accessible via PYTHONPATH
from ipd_visualization import visualize_all_results # Assuming ipd_visualization.py is accessible

#%%

def main():
    parser = argparse.ArgumentParser(description="Run IPD experiments with LLMs")
    parser.add_argument("--mode", choices=["run", "visualize", "both"], default="both",
                       help="Mode to run: experiments only, visualization only, or both")
    parser.add_argument("--output_dir", default="results", help="Directory to save results")
    parser.add_argument("--vis_dir", default="visualizations", help="Directory to save visualizations")
    parser.add_argument("--subset", action="store_true", help="Use a subset of strategies for testing")
    parser.add_argument("--turns", type=int, default=10, help="Number of turns in each match")
    parser.add_argument("--obfuscated-dir", required=True,
                       help="Directory containing pre-obfuscated strategy files (e.g., output from batch_obfuscate.py)")
    parser.add_argument("--refactored-dir", default="./refactored",
                       help="Directory containing original refactored strategy files.")
    parser.add_argument("--perturbations", nargs='+', default=["unmasked", "masked", "obfuscated"],
                       choices=["unmasked", "masked", "obfuscated"],
                       help="List of perturbation types to run (e.g., --perturbations unmasked masked). Default is all.")


    args = parser.parse_args()

    print("Loading and processing strategy program attributes...")
    try:
        from program_attribute import add_program_attributes
        add_program_attributes(
            refactored_dir=args.refactored_dir,
            obfuscated_dir=args.obfuscated_dir
        )
        print("Strategy program attributes loaded successfully.")
        if axl.all_strategies and hasattr(axl.all_strategies[0], 'program_obfuscated') and \
           getattr(axl.all_strategies[0], 'program_obfuscated') is None:
            print("Warning: 'program_obfuscated' attribute is None for the first strategy. Check paths and obfuscated files.")

    except ImportError:
        print("Error: program_attribute module not found. Cannot proceed without program attributes.")
        return 
    except Exception as e:
        print(f"Error: Failed during program attribute processing: {e}")
        return 
    
    MODEL_CONFIGS = [
        # FIRST RUN:
        #{
        #   "name": "Qwen/Qwen2.5-7B-Instruct",
        #   "api": "huggingface",
        #   "prompt_strategies": ["ZS", "COT"], 
        #   "display_name": "Qwen 2.5 (7B) (Instruct)" 
        #},
        #{
        #   "name": "gpt-4.1-2025-04-14",
        #   "api": "openai",
        #   "prompt_strategies": ["ZS", "COT"], 
        #   "display_name": "GPT-4.1" 
        #},
        # {
        #    "name": "deepseek-ai/DeepSeek-R1", # Example, replace with your actual model ID if different
        #    "api": "huggingface",
        #    "prompt_strategies": ["ZS"], 
        #    "display_name": "DeepSeek-R1" 
        # },
        #{
        #    "name": "deepseek-ai/DeepSeek-V3-0324", # Example, replace with your actual model ID if different
        #    "api": "huggingface",
        #    "prompt_strategies": ["COT"], 
        #    "display_name": "DeepSeek-V3-0324" 
        #}
        #{
        #   "name": "Qwen/Qwen2.5-Coder-32B-Instruct",
        #   "api": "huggingface",
        #   "prompt_strategies": ["ZS", "COT"], 
        #   "display_name": "Qwen 2.5 Coder (32B) (Instruction-Tuned)" 
        #},
        #{
        #   "name": "mistralai/Mistral-Small-24B-Instruct-2501",
        #   "api": "huggingface",
        #    "prompt_strategies": ["ZS", "COT"], 
        #    "display_name": "Mistral Small 24B (Instruction-Tuned)" 
        #},
        #{
        #   "name": "mistralai/Mixtral-8x7B-v0.1",
        #   "api": "huggingface",
        #   "prompt_strategies": ["ZS", "COT"], 
        #   "display_name": "Mixtral 8x7B" 
        #},
        #{
        #   "name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        #   "api": "huggingface",
        #   "prompt_strategies": ["ZS", "COT"], 
        #   "display_name": "Mixtral 8x7B (Instruction-Tuned)" 
        #},
        #{
        #   "name": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        #   "api": "huggingface",
        #   "prompt_strategies": ["ZS", "COT"], 
        #   "display_name": "Mixtral 8x22B (Instruction-Tuned)" 
        # },
        #{
        #   "name": "mistralai/Mistral-7B-Instruct-v0.3",
        #   "api": "huggingface",
        #   "prompt_strategies": ["ZS", "COT"], 
        #   "display_name": "Mistral 7B (Instruction-Tuned)" 
        #},
        #{
        #    "name": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        #    "api": "huggingface",
        #    "prompt_strategies": ["ZS", "COT"], 
        #    "display_name": "Nous Hermes 2" 
        #},
        #{
        #   "name": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        #   "api": "huggingface",
        #   "prompt_strategies": ["ZS", "COT"], 
        #   "display_name": "Llama 3.1 (70B) (NVIDIA)" 
        #},
    ]
    
    model_display_names = {config["name"]: config.get("display_name", config["name"]) 
                          for config in MODEL_CONFIGS}
        
    if args.subset:
        index_list = [191, 27, 38, 201, 76, 90, 93, 2, 34, 41, 103, 133] # Example subset (add) 
        AXELROD_STRATEGIES = [axl.all_strategies[i] for i in index_list if i < len(axl.all_strategies)]
        print(f"Using subset of {len(AXELROD_STRATEGIES)} strategies for testing")
    else:
        AXELROD_STRATEGIES = axl.all_strategies
        print(f"Using all {len(AXELROD_STRATEGIES)} strategies")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)
    
    if args.mode in ["run", "both"]:
        print(f"Running experiments with {len(MODEL_CONFIGS)} models.")
        print(f"Selected perturbations: {args.perturbations}")
        print(f"Results will be saved to: {args.output_dir}")
        
        run_experiment(
            model_configs=MODEL_CONFIGS,
            axelrod_strategies=AXELROD_STRATEGIES,
            num_turns=args.turns,
            output_dir=args.output_dir,
            perturbations_to_run=args.perturbations
        )
    
    if args.mode in ["visualize", "both"]:
        print(f"Generating visualizations from results in: {args.output_dir}")
        print(f"Visualizations will be saved to: {args.vis_dir}")
        visualize_all_results(
            results_dir=args.output_dir,
            output_dir=args.vis_dir,
            model_configs=MODEL_CONFIGS,
            model_display_names=model_display_names 
        )
    
    print("All tasks completed successfully!")


if __name__ == "__main__":
    main()
