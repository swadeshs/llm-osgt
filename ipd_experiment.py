DEBUG_TRACES = False

import json
import os
import sys
import time
from pathlib import Path
from typing import Literal, Optional, Dict, List, Any
import math
import re
import pandas as pd
import random
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import axelrod as axl
C, D = axl.Action.C, axl.Action.D
from openai import OpenAI

load_dotenv()

class LLMAPI:
    def __init__(self, model_name="gpt-4o-mini-2024-07-18", api="openai", prompts=None, model_configs=None):
        self.model_name = model_name
        self.api = api
        self.prompts = prompts
        self.model_configs = model_configs or {}
        
        if "deepseek-r1" in model_name.lower():
            self.model_configs["temperature"] = 0.6
            self.model_configs["use_system_prompt"] = False
            self.model_configs["max_tokens"] = 8192
        
        if "mistral-small" in model_name.lower():
            self.model_configs["temperature"] = 0.15
        
        if api == "huggingface":
            try:
                from huggingface_hub import InferenceClient
                api_token = os.getenv("HF_API_KEY")
                
                if not api_token:
                    print("WARNING: API key not found in environment variables.")

                self.client = InferenceClient(
                    model=self.model_name,
                    token=api_token,
                    provider="novita"
                )
            except ImportError:
                print("Error: huggingface_hub package not installed.")
                raise
        elif api == "openai":
            try:
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                self.client = client.chat.completions
            except Exception as e:
                print(f"Error initializing OpenAI client: {e}")
                raise
        elif api == "google":
            try:
                import google.generativeai as genai
                google_api_key = os.getenv("GOOGLE_API_KEY")
                if not google_api_key:
                    print("WARNING: GOOGLE_API_KEY not found in environment variables")
                genai.configure(api_key=google_api_key)
                self.client = genai.GenerativeModel(model_name)
            except ImportError:
                print("Error: google-generativeai package not installed.")
                raise
        else:
            raise ValueError(f"Unsupported API: {api}")
    
    def _parse_prediction(self, response_text: str, max_tokens: Optional[int] = None) -> str:
        prediction = "N/A" 
        if not response_text or not isinstance(response_text, str):
            return prediction 

        text_lower = response_text.lower()
        yes_pattern = r'(\byes\b|["\']yes["\']|prediction: yes|answer: yes|^yes[.,!?\s]*$)'
        no_pattern = r'(\bno\b|["\']no["\']|prediction: no|answer: no|^no[.,!?\s]*$)'
        yes_matches = list(re.finditer(yes_pattern, text_lower, re.MULTILINE))
        no_matches = list(re.finditer(no_pattern, text_lower, re.MULTILINE))

        if yes_matches and not no_matches: prediction = "yes"
        elif no_matches and not yes_matches: prediction = "no"
        elif yes_matches and no_matches:
            prediction = "N/A_Ambiguous"
        else:
            words = response_text.split()
            if words:
                last_word_cleaned = words[-1].strip().rstrip('.,!?;:"\'').lower()
                last_word_cleaned = re.sub(r'[^a-z]', '', last_word_cleaned) 
                if last_word_cleaned == "yes": prediction = "yes"
                elif last_word_cleaned == "no": prediction = "no"
            else:
                prediction = "N/A_Empty_Split"

        is_truncated = False
        if max_tokens is not None:
            is_truncated = response_text.endswith("...") or len(response_text) >= (max_tokens - 50)

        if is_truncated and prediction == "N/A":
            prediction = "N/A_Truncated_No_Signal"
        return prediction

    def query_llm(self, prompt_type, opponent_strategy):
        if self.api == "huggingface":
            return self._hf_query_llm(prompt_type, opponent_strategy)
        elif self.api == "openai":
            return self._oai_query_llm(prompt_type, opponent_strategy)
        elif self.api == "google":
            return self._google_query_llm(prompt_type, opponent_strategy)
        return "API Type Not Supported", "N/A_Unsupported_API"
    
    def _hf_query_llm(self, prompt_type, opponent_strategy):
        try:
            prompt_template = self.prompts[prompt_type]
            messages = []
            trace = None

            use_system = self.model_configs.get("use_system_prompt", True)
            if prompt_template[0]["role"] == "system" and use_system:
                messages.append(prompt_template[0])
                start_idx = 1
            else:
                start_idx = 0

            for message in prompt_template[start_idx:-1]:
                messages.append(message)

            messages.append({
                "role": "user",
                "content": f"This is your opponent's strategy program:\n{opponent_strategy}"
            })

            max_tokens = self.model_configs.get("max_tokens", 4096)
            temperature = self.model_configs.get("temperature", 0.0)

            if prompt_type not in ("ZS", "FS"):
                try:
                    trace_response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    trace = trace_response.choices[0].message.content.strip()
                    messages.append({"role": "assistant", "content": trace})

                except Exception as e:
                    error_trace = f"Error during trace generation: {str(e)[:150]}..."
                    return error_trace, "N/A"

                messages.append(prompt_template[-1])

                try:
                    prediction_response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=self.model_configs.get("prediction_max_tokens", 50),
                        temperature=temperature
                    )
                    pred_unprocessed = prediction_response.choices[0].message.content.strip()

                except Exception as e:
                    return trace, "N/A"

            else:
                messages.append(prompt_template[-1])

                try:
                    prediction_response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    pred_unprocessed = prediction_response.choices[0].message.content.strip()
                    trace = pred_unprocessed

                except Exception as e:
                    error_trace = f"Error during ZS/FS generation: {str(e)[:150]}..."
                    return error_trace, "N/A"

            words = pred_unprocessed.split()
            if not words:
                prediction = "N/A"
            else:
                last_word_cleaned = words[-1].strip().rstrip('.,!?;:"\'').lower()
                prediction = re.sub(r'[^a-z]', '', last_word_cleaned)

            return trace, prediction

        except Exception as e:
            return f"General Error: {str(e)[:100]}...", "N/A"
    
    def _oai_query_llm(self, prompt_type, opponent_strategy):
        try:
            prompt = self.prompts[prompt_type]
            messages = []
            for message in prompt[:-1]:
                messages.append(message)
            messages.append({"role": "user", "content": "Opponent's strategy:\n" + opponent_strategy})

            trace = None
            is_o_series_model = "o4-mini" in self.model_name.lower()
            
            api_params_base = {
                "model": self.model_name,
                "n": 1,
                "stop": None,
            }

            if not is_o_series_model:
                api_params_base["temperature"] = 0.0

            if prompt_type not in ("ZS", "FS"):
                query_params_cot = {**api_params_base, "messages": list(messages)}
                query = self.client.create(**query_params_cot)
                trace = query.choices[0].message.content.strip()
                messages.append({"role": "assistant", "content": trace})

            messages.append(prompt[-1])
            query_params_final = {**api_params_base, "messages": list(messages)}
            query = self.client.create(**query_params_final)
            pred_unprocessed = query.choices[0].message.content.strip()

            prediction = self._parse_prediction(pred_unprocessed)
            if not pred_unprocessed.strip() and prediction == "N/A":
                prediction = "N/A_Empty_API_Response"

            return (trace if trace is not None else pred_unprocessed), prediction

        except Exception as e:
            error_details = str(e)
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                error_details = f"{str(e)} - API Response: {e.response.text[:500]}"
            elif hasattr(e, 'json_body') and e.json_body and 'error' in e.json_body:
                error_details = f"{str(e)} - API Error: {e.json_body['error'].get('message', e.json_body['error'])}"

            error_msg = f"OpenAI Error ({self.model_name}, {prompt_type}): {type(e).__name__} - {error_details[:200]}"
            return error_msg, "N/A_OpenAI_Error"
    
    def _google_query_llm(self, prompt_type, opponent_strategy):
        try:
            prompt = self.prompts[prompt_type]
            messages = []
            system_content = None
            if prompt[0]["role"] == "system":
                system_content = prompt[0]["content"]
                start_idx = 1
            else:
                start_idx = 0
            
            current_user_parts = []
            if system_content:
                current_user_parts.append(f"System instruction: {system_content}")

            for msg_idx in range(start_idx, len(prompt) -1):
                message = prompt[msg_idx]
                if message["role"] == "user":
                    if current_user_parts:
                         current_user_parts.append(message["content"])
                    else:
                         current_user_parts = [message["content"]]
                elif message["role"] == "assistant":
                    if current_user_parts:
                        messages.append({"role": "user", "parts": current_user_parts})
                        current_user_parts = []
                    messages.append({"role": "model", "parts": [message["content"]]})
            
            if current_user_parts:
                messages.append({"role": "user", "parts": current_user_parts})
            
            messages.append({"role": "user", "parts": [f"Opponent's strategy:\n{opponent_strategy}"]})
            
            trace = None
            if prompt_type not in ("ZS", "FS"):
                response = self.client.generate_content(messages)
                trace = response.text.strip()
                messages.append({"role": "model", "parts": [trace]})
            
            final_prompt_message = prompt[-1]
            if final_prompt_message["role"] == "user":
                 messages.append({"role": "user", "parts": [final_prompt_message["content"]]})
            else:
                 messages.append({"role": "model", "parts": [final_prompt_message["content"]]})

            response = self.client.generate_content(messages)
            pred_unprocessed = response.text.strip()
            prediction = self._parse_prediction(pred_unprocessed)
            if not pred_unprocessed.strip() and prediction == "N/A":
                prediction = "N/A_Empty_API_Response"
            return (trace if trace is not None else pred_unprocessed), prediction
        except Exception as e:
            error_msg = f"Google Gemini Error ({self.model_name}, {prompt_type}): {type(e).__name__} - {str(e)[:100]}"
            return error_msg, "N/A_Google_Error"

class LiteratePrisoner(axl.Player):
    name = "LiteratePrisoner"
    classifier = {"memory_depth": float("inf"), "stochastic": True, "makes_use_of": ["opponent_history", "opponent_source"], "long_run_time": True, "inspects_source": True, "manipulates_source": False, "manipulates_state": False}

    def __init__(self, strategy_name="LiteratePrisoner", prompt_strategy="ZS", llm_api=None, use_masked=False, use_obf=False):
        super().__init__()
        self.llm_api = llm_api or LLMAPI(prompts=prompts_library)
        self.prompt_strategy = prompt_strategy
        self.strategy_name = strategy_name 
        self.initial_prediction = None
        self.initial_reasoning = None
        self.use_masked = use_masked
        self.use_obf = use_obf

    def strategy(self, opponent: axl.Player) -> axl.Action:
        if self.initial_prediction is None:
            self.initial_reasoning, self.initial_prediction = self.get_prediction(opponent, self.prompt_strategy, self.use_masked, self.use_obf)
        if self.initial_prediction == "yes": return axl.Action.C
        elif self.initial_prediction == "no": return axl.Action.D
        else: return axl.Action.D

    def get_prediction(self, opponent: axl.Player, prompt_strategy, use_masked=False, use_obf=False):
        program = "# Strategy code unavailable"
        try:
            if use_obf: program = opponent.program_obfuscated
            elif use_masked: program = opponent.program_masked
            else: program = opponent.program
        except AttributeError as e:
            program = f"# Attr error for {opponent.name if hasattr(opponent, 'name') else 'Unknown Opponent'}: {e}"
        except Exception as e:
            program = f"# Generic error retrieving program: {e}"
        trace, prediction = self.llm_api.query_llm(prompt_strategy, program)
        return trace, prediction

def verify_program_obfuscation(strategy_instance, strategy_name):
    if not hasattr(strategy_instance, 'program'): return False
    if not hasattr(strategy_instance, 'program_masked'): return False
    if not hasattr(strategy_instance, 'program_obfuscated'): return False
    return True

def analyze_strategies(model_name, api_type, prompt_strategy, axelrod_strategies,
                      num_turns, use_masked=False, use_obf=False, program_modification=None):
    llm_api = LLMAPI(model_name=model_name, api=api_type, prompts=prompts_library[program_modification])
    data_cooperator = []
    MAX_RETRIES = 3

    program_type_str = "obfuscated" if use_obf else ("masked" if use_masked else "unmasked")

    STRICT_API_ERROR_TRACE_SIGNATURES = [
        "N/A_HF_Trace_Error", "N/A_HF_Pred_Error", "N/A_HF_ZSFS_Error", "N/A_HF_General_Error",
        "N/A_OpenAI_Error", "N/A_Google_Error", "N/A_Unsupported_API",
        "Huggingface Error", "OpenAI Error", "Google Gemini Error",
        "Rate limit exceeded", "model service overloaded", "API key invalid",
        "upstream service error", "content generation failed due to internal error",
        "could not connect to host", "timeout", "service unavailable", "internal server error",
        "authentication failed", "permission denied", "invalid request"
    ]
    VALID_PREDICTIONS = ["yes", "no"]

    for strategy_class in axelrod_strategies:
        strategy_name = strategy_class().name
        strategy_instance_for_attrs = strategy_class()
        is_stochastic = strategy_instance_for_attrs.classifier.get('stochastic', False)

        try:
            if use_obf: strategy_program = strategy_instance_for_attrs.program_obfuscated
            elif use_masked: strategy_program = strategy_instance_for_attrs.program_masked
            else: strategy_program = strategy_instance_for_attrs.program
        except AttributeError:
            strategy_program = "# Program attribute missing"
            data_cooperator.append({
                "Model": model_name, "API": api_type, "Prompt Strategy": prompt_strategy,
                "Opponent Strategy": strategy_name, "Stochastic": is_stochastic,
                "Trace": "Program Attribute Missing", "Prediction": "N/A_Attr_Error",
                "Ground Truth": "N/A", "Correct Prediction": "N/A",
                "Opponent Actions": "N/A", "Opponent Payoffs": "N/A",
                "Cooperator Payoffs": "N/A", "Program": strategy_program,
                "Program_Type": program_type_str, "API_Failure": True
            })
            continue

        api_failure_for_csv = False
        trace_for_csv, prediction_for_csv = "Not Set", "Not Set"
        ground_truth_cooperator, correct_prediction_cooperator = "Not Set", "Not Set"
        opponent_actions_cooperator, opponent_payoffs_cooperator, cooperator_payoffs = "Not Set", "Not Set", "Not Set"

        literate_player = LiteratePrisoner(
            prompt_strategy=prompt_strategy, llm_api=llm_api,
            use_masked=use_masked, use_obf=use_obf
        )

        for attempt in range(MAX_RETRIES):
            current_trace, current_prediction = "Retry Error", "N/A_Retry_Error_Default"
            retry_needed = False
            error_reason_summary = []

            try:
                current_trace, current_prediction = literate_player.get_prediction(
                    strategy_class(), prompt_strategy, use_masked, use_obf
                )

                if current_trace is None:
                    retry_needed = True
                    error_reason_summary.append("Trace is None")
                if current_prediction is None:
                    retry_needed = True
                    error_reason_summary.append("Prediction is None")

                if isinstance(current_trace, str):
                    for sig in STRICT_API_ERROR_TRACE_SIGNATURES:
                        if sig.lower() in current_trace.lower():
                            retry_needed = True
                            error_reason_summary.append(f"Strict API Error Signature in Trace ('{sig}')")
                            break

                if not isinstance(current_prediction, str) or current_prediction not in VALID_PREDICTIONS:
                    retry_needed = True
                    error_reason_summary.append(f"Invalid Prediction ('{str(current_prediction)}')")

                if not retry_needed:
                    trace_for_csv, prediction_for_csv = current_trace, current_prediction
                    api_failure_for_csv = False
                    break
                else:
                    reason_str = "; ".join(list(set(error_reason_summary)))
                    trace_for_csv = str(current_trace) if current_trace is not None else "Trace_None_Before_Retry"
                    prediction_for_csv = str(current_prediction) if current_prediction is not None else "Pred_None_Before_Retry"

                    if attempt < MAX_RETRIES - 1:
                        time.sleep(2 * (attempt + 1))
                    else:
                        api_failure_for_csv = True
                        break

            except Exception as e_call:
                trace_for_csv = f"Outer Exception: {type(e_call).__name__} - {str(e_call)[:150]}"
                prediction_for_csv = "N/A_Outer_Exception"
                api_failure_for_csv = True

                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    break

        if api_failure_for_csv:
            ground_truth_cooperator, correct_prediction_cooperator = "API Failed", "API Failed"
            opponent_actions_cooperator, opponent_payoffs_cooperator, cooperator_payoffs = "API Failed", "API Failed", "API Failed"
        else:
            try:
                cooperator = axl.Cooperator()
                match_coop = axl.Match((strategy_class(), cooperator), turns=num_turns)
                match_coop.play()

                opponent_actions_cooperator = [str(item[0]) for item in match_coop.result]
                opponent_payoffs_cooperator = [int(item[0]) for item in match_coop.scores()]
                cooperator_payoffs = [int(item[1]) for item in match_coop.scores()]

                opponent_always_cooperates_coop = all(action == axl.Action.C for action in [r[0] for r in match_coop.result])
                ground_truth_cooperator = "yes" if opponent_always_cooperates_coop else "no"
                correct_prediction_cooperator = "yes" if prediction_for_csv == ground_truth_cooperator else "no"
            except Exception as e_analysis:
                ground_truth_cooperator, correct_prediction_cooperator = "Analysis Failed", "Analysis Failed"
                opponent_actions_cooperator, opponent_payoffs_cooperator, cooperator_payoffs = "Analysis Failed", "Analysis Failed", "Analysis Failed"

        data_cooperator.append({
            "Model": model_name, "API": api_type, "Prompt Strategy": prompt_strategy,
            "Opponent Strategy": strategy_name, "Stochastic": is_stochastic,
            "Trace": trace_for_csv, "Prediction": prediction_for_csv,
            "Ground Truth": ground_truth_cooperator, "Correct Prediction": correct_prediction_cooperator,
            "Opponent Actions": opponent_actions_cooperator, "Opponent Payoffs": opponent_payoffs_cooperator,
            "Cooperator Payoffs": cooperator_payoffs, "Program": strategy_program,
            "Program_Type": program_type_str, "API_Failure": api_failure_for_csv
        })
    return pd.DataFrame(data_cooperator)

def run_experiment(model_configs, axelrod_strategies, num_turns, output_dir, perturbations_to_run=None):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    if perturbations_to_run is None:
        perturbations_to_run = ["unmasked", "masked", "obfuscated"]

    for model_config in model_configs:
        model_name = model_config['name']
        api_type = model_config['api']
        prompt_strategies_for_model = model_config.get('prompt_strategies', ["ZS"]) 
        if not isinstance(prompt_strategies_for_model, list):
            prompt_strategies_for_model = [prompt_strategies_for_model]

        model_dir = os.path.join(output_dir, model_name.replace("/", "_").replace(":", "_"))
        os.makedirs(model_dir, exist_ok=True)
        
        for prompt_strategy in prompt_strategies_for_model:
            prompt_dir = os.path.join(model_dir, prompt_strategy)
            os.makedirs(prompt_dir, exist_ok=True)
            
            all_program_modifications = {
                "unmasked": {"use_masked": False, "use_obf": False, "mod_key": "vanilla"},
                "masked": {"use_masked": True, "use_obf": False, "mod_key": "masked"},
                "obfuscated": {"use_masked": False, "use_obf": True, "mod_key": "obfuscated"}
            }

            for prog_type_key in perturbations_to_run:
                if prog_type_key not in all_program_modifications:
                    continue
                
                prog_params = all_program_modifications[prog_type_key]
                
                df_results = analyze_strategies(
                    model_name, api_type, prompt_strategy, 
                    axelrod_strategies, num_turns, 
                    use_masked=prog_params["use_masked"], use_obf=prog_params["use_obf"],
                    program_modification=prog_params["mod_key"]
                )
                
                if not df_results.empty:
                    csv_path = os.path.join(prompt_dir, f"{prompt_strategy}_{prog_type_key}.csv")
                    df_results.to_csv(csv_path, index=False)
                    print(f"{prog_type_key.capitalize()} results saved to {csv_path}")
                else:
                    print(f"No results generated for {prog_type_key} programs with {prompt_strategy} for {model_name}. CSV not saved.")


    print("\nExperiment data collection completed successfully!")

#%%

# Prompt strategies library
prompts_library = {
    "vanilla": {"ZS": [
        {
            "role": "system",
            "content": "You are an expert at game theory, specifically the Iterated Prisoner's Dilemma (IPD). Your specialty is analyzing Python code to accurately predict its behavior step-by-step. You excel at tracing execution flows through inheritance hierarchies, understanding class relationships, and detecting subtle patterns in decision-making algorithms. You have particular expertise in identifying when a strategy will always cooperate versus when it might defect under certain conditions, especially when analyzing stochastic elements."
        },
        {
            "role": "user",
            "content": """You are an LLM agent preparing to play an Iterated Prisoner's Dilemma (IPD) for {num_turns} rounds. You are given the source code of your opponent's strategy program and are tasked with predicting your opponent's behavior.

Your objective is to analyze your opponent's strategy program and predict whether your opponent will cooperate (play C) against you for all {num_turns} rounds, assuming you play a purely cooperative strategy (play C for all {num_turns} rounds).

IMPORTANT: If the code contains multiple classes, focus on the final subclass mentioned in the program. However, remember that subclasses inherit strategies and initialization parameters from parent classes (and sometimes override them). You must analyze the entire inheritance hierarchy to understand the complete strategy implementation."""
            .format(num_turns=10) 
        },
        {
            "role": "user",
            "content": """Assuming you cooperate (play C) for all {num_turns} rounds, predict whether or not your opponent will cooperate (play C) in all {num_turns} rounds against you. Respond with only 'yes' or 'no', with no further formatting."""
            .format(num_turns=10)
        }
    ],
    "COT": [
        {
            "role": "system",
            "content": "You are an expert at game theory, specifically the Iterated Prisoner's Dilemma (IPD). Your specialty is analyzing Python code to accurately predict its behavior step-by-step. You excel at tracing execution flows through inheritance hierarchies, understanding class relationships, and detecting subtle patterns in decision-making algorithms. You have particular expertise in identifying when a strategy will always cooperate versus when it might defect under certain conditions, especially when analyzing stochastic elements."
        },
        {
            "role": "user",
            "content": """You are an LLM agent preparing to play an Iterated Prisoner's Dilemma (IPD) for {num_turns} rounds. You are given the source code of your opponent's strategy program and are tasked with predicting your opponent's behavior.

Your objective is to analyze your opponent's strategy program and predict whether your opponent will cooperate (play C) against you for all {num_turns} rounds, assuming you play a purely cooperative strategy (play C for all {num_turns} rounds).

IMPORTANT: If the code contains multiple classes, focus on the final subclass mentioned in the program. However, remember that subclasses inherit strategies and initialization parameters from parent classes (and sometimes override them). You must analyze the entire inheritance hierarchy to understand the complete strategy implementation."""
            .format(num_turns=10)
        },
        {
            "role": "user",
            "content": """Assume you will cooperate (play C) for all {num_turns} rounds, analyze your opponent's strategy carefully, think step by step, and clearly articulate your reasoning to predict if this strategy will cooperate (play C) in all {num_turns} rounds against you."""
            .format(num_turns=10)
        },
        {
            "role": "user",
            "content": """Based on your reasoning, assuming you cooperate (play C) for all {num_turns} rounds, predict whether or not your opponent will cooperate (play C) in all {num_turns} rounds against you. Respond with only 'yes' or 'no', with no further formatting."""
            .format(num_turns=10)
        }
    ],
    "COT-Unrolled": [
        {
            "role": "system",
            "content": "You are an expert at game theory, specifically the Iterated Prisoner's Dilemma (IPD). Your specialty is analyzing Python code to accurately predict its behavior step-by-step. You excel at tracing execution flows through inheritance hierarchies, understanding class relationships, and detecting subtle patterns in decision-making algorithms. You have particular expertise in identifying when a strategy will always cooperate versus when it might defect under certain conditions, especially when analyzing stochastic elements."
        },
        {
            "role": "user",
            "content": """You are an LLM agent preparing to play an Iterated Prisoner's Dilemma (IPD) for {num_turns} rounds. You are given the source code of your opponent's strategy program and are tasked with predicting your opponent's behavior.

Your objective is to analyze your opponent's strategy program and predict whether your opponent will cooperate (play C) against you for all {num_turns} rounds, assuming you play a purely cooperative strategy (play C for all {num_turns} rounds).

IMPORTANT: If the code contains multiple classes, focus on the final subclass mentioned in the program. However, remember that subclasses inherit strategies and initialization parameters from parent classes (and sometimes override them). You must analyze the entire inheritance hierarchy to understand the complete strategy implementation."""
            .format(num_turns=10)
        },
        {
            "role": "user",
            "content": """Assume you will cooperate (play C) for all {num_turns} rounds, analyze your opponent's strategy carefully, think step by step, clearly articulate your reasoning, and predict what moves your opponent will play against you for each of the {num_turns} rounds. At the end of your analysis, state a list of predictions of your opponent's behavior (e.g. [C, C, C, C...], [D, D, D, D...] or similar) for each round."""
            .format(num_turns=10)            
        },
        {
            "role": "user",
            "content": """Based on your reasoning, assuming you cooperate (play C) for all {num_turns} rounds, predict whether or not your opponent will cooperate (play C) in all {num_turns} rounds against you. Respond with only 'yes' or 'no', with no further formatting."""
            .format(num_turns=10)
        }
    ]}, 
    "masked": {"ZS": [ 
        {
            "role": "system",
            "content": "You are an expert at game theory, specifically the Iterated Prisoner's Dilemma (IPD). Your specialty is analyzing Python code to accurately predict its behavior step-by-step. You excel at tracing execution flows through inheritance hierarchies, understanding class relationships, and detecting subtle patterns in decision-making algorithms. You have particular expertise in identifying when a strategy will always cooperate versus when it might defect under certain conditions, especially when analyzing stochastic elements."
        },
        {
            "role": "user",
            "content": """You are an LLM agent preparing to play an Iterated Prisoner's Dilemma (IPD) for {num_turns} rounds. You are given the source code of your opponent's strategy program and are tasked with predicting your opponent's behavior.

Your objective is to analyze your opponent's strategy program and predict whether your opponent will cooperate (play C) against you for all {num_turns} rounds, assuming you play a purely cooperative strategy (play C for all {num_turns} rounds).

IMPORTANT: If the code contains multiple classes, focus on the final subclass mentioned in the program. However, remember that subclasses inherit strategies and initialization parameters from parent classes (and sometimes override them). You must analyze the entire inheritance hierarchy to understand the complete strategy implementation."""
            .format(num_turns=10)
        },
        {
            "role": "user",
            "content": """Assuming you cooperate (play C) for all {num_turns} rounds, predict whether or not your opponent will cooperate (play C) in all {num_turns} rounds against you. Respond with only 'yes' or 'no', with no further formatting."""
            .format(num_turns=10)
        }
    ],
    "COT": [ 
        {
            "role": "system",
            "content": "You are an expert at game theory, specifically the Iterated Prisoner's Dilemma (IPD). Your specialty is analyzing Python code to accurately predict its behavior step-by-step. You excel at tracing execution flows through inheritance hierarchies, understanding class relationships, and detecting subtle patterns in decision-making algorithms. You have particular expertise in identifying when a strategy will always cooperate versus when it might defect under certain conditions, especially when analyzing stochastic elements."
        },
        {
            "role": "user",
            "content": """You are an LLM agent preparing to play an Iterated Prisoner's Dilemma (IPD) for {num_turns} rounds. You are given the source code of your opponent's strategy program and are tasked with predicting your opponent's behavior.

Your objective is to analyze your opponent's strategy program and predict whether your opponent will cooperate (play C) against you for all {num_turns} rounds, assuming you play a purely cooperative strategy (play C for all {num_turns} rounds).

IMPORTANT: If the code contains multiple classes, focus on the final subclass mentioned in the program. However, remember that subclasses inherit strategies and initialization parameters from parent classes (and sometimes override them). You must analyze the entire inheritance hierarchy to understand the complete strategy implementation."""
            .format(num_turns=10)
        },
        {
            "role": "user",
            "content": """Assume you will cooperate (play C) for all {num_turns} rounds, analyze your opponent's strategy carefully, think step by step, and clearly articulate your reasoning to predict if this strategy will cooperate (play C) in all {num_turns} rounds against you."""
            .format(num_turns=10)
        },
        {
            "role": "user",
            "content": """Based on your reasoning, assuming you cooperate (play C) for all {num_turns} rounds, predict whether or not your opponent will cooperate (play C) in all {num_turns} rounds against you. Respond with only 'yes' or 'no', with no further formatting."""
            .format(num_turns=10)
        }
    ],
    "COT-Unrolled": [ 
        {
            "role": "system",
            "content": "You are an expert at game theory, specifically the Iterated Prisoner's Dilemma (IPD). Your specialty is analyzing Python code to accurately predict its behavior step-by-step. You excel at tracing execution flows through inheritance hierarchies, understanding class relationships, and detecting subtle patterns in decision-making algorithms. You have particular expertise in identifying when a strategy will always cooperate versus when it might defect under certain conditions, especially when analyzing stochastic elements."
        },
        {
            "role": "user",
            "content": """You are an LLM agent preparing to play an Iterated Prisoner's Dilemma (IPD) for {num_turns} rounds. You are given the source code of your opponent's strategy program and are tasked with predicting your opponent's behavior.

Your objective is to analyze your opponent's strategy program and predict whether your opponent will cooperate (play C) against you for all {num_turns} rounds, assuming you play a purely cooperative strategy (play C for all {num_turns} rounds).

IMPORTANT: If the code contains multiple classes, focus on the final subclass mentioned in the program. However, remember that subclasses inherit strategies and initialization parameters from parent classes (and sometimes override them). You must analyze the entire inheritance hierarchy to understand the complete strategy implementation."""
            .format(num_turns=10)
        },
        {
            "role": "user",
            "content": """Assume you will cooperate (play C) for all {num_turns} rounds, analyze your opponent's strategy carefully, think step by step, clearly articulate your reasoning, and predict what moves your opponent will play against you for each of the {num_turns} rounds. At the end of your analysis, state a list of predictions of your opponent's behavior (e.g. [C, C, C, C...], [D, D, D, D...] or similar) for each round."""
            .format(num_turns=10)            
        },
        {
            "role": "user",
            "content": """Based on your reasoning, assuming you cooperate (play C) for all {num_turns} rounds, predict whether or not your opponent will cooperate (play C) in all {num_turns} rounds against you. Respond with only 'yes' or 'no', with no further formatting."""
            .format(num_turns=10)
        }
    ]},
    "obfuscated": {"ZS": [ 
        {
            "role": "system",
            "content": "You are an expert at game theory, specifically the Iterated Prisoner's Dilemma (IPD). Your specialty is analyzing Python code to accurately predict its behavior step-by-step. You excel at tracing execution flows through inheritance hierarchies, understanding class relationships, and detecting subtle patterns in decision-making algorithms. You have particular expertise in identifying when a strategy will always cooperate versus when it might defect under certain conditions, especially when analyzing stochastic elements."
        },
        {
            "role": "user",
            "content": """You are an LLM agent preparing to play an Iterated Prisoner's Dilemma (IPD) for {num_turns} rounds. You are given the source code of your opponent's strategy program and are tasked with predicting your opponent's behavior.

Your objective is to analyze your opponent's strategy program and predict whether your opponent will cooperate (play C) against you for all {num_turns} rounds, assuming you play a purely cooperative strategy (play C for all {num_turns} rounds).

Note: This code has been obfuscated through identifier renaming. Focus on understanding the code's control flow, structures, and the relationships between different parts of the code to reason about its behavior. Do not rely on the names of variables, functions, or classes to understand their purpose, as these names are meaningless and randomly generated. The underlying logic and structure of the code remain the same.

IMPORTANT: If the code contains multiple classes, focus on the final subclass mentioned in the program. However, remember that subclasses inherit strategies and initialization parameters from parent classes (and sometimes override them). You must analyze the entire inheritance hierarchy to understand the complete strategy implementation."""
            .format(num_turns=10)
        },
        {
            "role": "user",
            "content": """Assuming you cooperate (play C) for all {num_turns} rounds, predict whether or not your opponent will cooperate (play C) in all {num_turns} rounds against you. Respond with only 'yes' or 'no', with no further formatting."""
            .format(num_turns=10)
        }
    ],
    "COT": [ 
        {
            "role": "system",
            "content": "You are an expert at game theory, specifically the Iterated Prisoner's Dilemma (IPD). Your specialty is analyzing Python code to accurately predict its behavior step-by-step. You excel at tracing execution flows through inheritance hierarchies, understanding class relationships, and detecting subtle patterns in decision-making algorithms. You have particular expertise in identifying when a strategy will always cooperate versus when it might defect under certain conditions, especially when analyzing stochastic elements."
        },
        {
            "role": "user",
            "content": """You are an LLM agent preparing to play an Iterated Prisoner's Dilemma (IPD) for {num_turns} rounds. You are given the source code of your opponent's strategy program and are tasked with predicting your opponent's behavior.

Your objective is to analyze your opponent's strategy program and predict whether your opponent will cooperate (play C) against you for all {num_turns} rounds, assuming you play a purely cooperative strategy (play C for all {num_turns} rounds).

Note: This code has been obfuscated through identifier renaming. Focus on understanding the code's control flow, structures, and the relationships between different parts of the code to reason about its behavior. Do not rely on the names of variables, functions, or classes to understand their purpose, as these names are meaningless and randomly generated. The underlying logic and structure of the code remain the same.

IMPORTANT: If the code contains multiple classes, focus on the final subclass mentioned in the program. However, remember that subclasses inherit strategies and initialization parameters from parent classes (and sometimes override them). You must analyze the entire inheritance hierarchy to understand the complete strategy implementation."""
            .format(num_turns=10)
        },
        {
            "role": "user",
            "content": """Assume you will cooperate (play C) for all {num_turns} rounds, analyze your opponent's strategy carefully, think step by step, and clearly articulate your reasoning to predict if this strategy will cooperate (play C) in all {num_turns} rounds against you."""
            .format(num_turns=10)
        },
        {
            "role": "user",
            "content": """Based on your reasoning, assuming you cooperate (play C) for all {num_turns} rounds, predict whether or not your opponent will cooperate (play C) in all {num_turns} rounds against you. Respond with only 'yes' or 'no', with no further formatting."""
            .format(num_turns=10)
        }
    ],
    "COT-Unrolled": [ 
        {
            "role": "system",
            "content": "You are an expert at game theory, specifically the Iterated Prisoner's Dilemma (IPD). Your specialty is analyzing Python code to accurately predict its behavior step-by-step. You excel at tracing execution flows through inheritance hierarchies, understanding class relationships, and detecting subtle patterns in decision-making algorithms. You have particular expertise in identifying when a strategy will always cooperate versus when it might defect under certain conditions, especially when analyzing stochastic elements."
        },
        {
            "role": "user",
            "content": """You are an LLM agent preparing to play an Iterated Prisoner's Dilemma (IPD) for {num_turns} rounds. You are given the source code of your opponent's strategy program and are tasked with predicting your opponent's behavior.

Your objective is to analyze your opponent's strategy program and predict whether your opponent will cooperate (play C) against you for all {num_turns} rounds, assuming you play a purely cooperative strategy (play C for all {num_turns} rounds).

Note: This code has been obfuscated through identifier renaming. Focus on understanding the code's control flow, structures, and the relationships between different parts of the code to reason about its behavior. Do not rely on the names of variables, functions, or classes to understand their purpose, as these names are meaningless and randomly generated. The underlying logic and structure of the code remain the same.

IMPORTANT: If the code contains multiple classes, focus on the final subclass mentioned in the program. However, remember that subclasses inherit strategies and initialization parameters from parent classes (and sometimes override them). You must analyze the entire inheritance hierarchy to understand the complete strategy implementation."""
            .format(num_turns=10)
        },
        {
            "role": "user",
            "content": """Assume you will cooperate (play C) for all {num_turns} rounds, analyze your opponent's strategy carefully, think step by step, clearly articulate your reasoning, and predict what moves your opponent will play against you for each of the {num_turns} rounds. At the end of your analysis, state a list of predictions of your opponent's behavior (e.g. [C, C, C, C...], [D, D, D, D...] or similar) for each round."""
            .format(num_turns=10)            
        },
        {
            "role": "user",
            "content": """Based on your reasoning, assuming you cooperate (play C) for all {num_turns} rounds, predict whether or not your opponent will cooperate (play C) in all {num_turns} rounds against you. Respond with only 'yes' or 'no', with no further formatting."""
            .format(num_turns=10)
        }
    ]}
}

#%%
if __name__ == "__main__":
    MODEL_CONFIGS_EXAMPLE = [
        {"name": "mistralai/Mistral-7B-Instruct-v0.1", "api": "huggingface", "prompt_strategies": ["ZS"]},
    ]
    NUM_TURNS_EXAMPLE = 5
    try:
        from program_attribute import add_program_attributes
    except ImportError: pass
    except FileNotFoundError: pass

    AXELROD_STRATEGIES_EXAMPLE = [axl.Cooperator(), axl.Defector(), axl.TitForTat()] 
    OUTPUT_DIR_EXAMPLE = "results_test_ipd_experiment_perturb"
    PERTURBATIONS_TO_RUN_EXAMPLE = ["unmasked", "masked"]

    run_experiment(
        model_configs=MODEL_CONFIGS_EXAMPLE, 
        axelrod_strategies=AXELROD_STRATEGIES_EXAMPLE, 
        num_turns=NUM_TURNS_EXAMPLE, 
        output_dir=OUTPUT_DIR_EXAMPLE,
        perturbations_to_run=PERTURBATIONS_TO_RUN_EXAMPLE
    )

