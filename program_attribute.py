#%%
import os
import sys
import re
import subprocess
import tempfile
import shutil
import axelrod as axl

def strip_comments_and_docstrings(source_code):
    """
    Remove all comments and docstrings from Python source code using regex.

    Args:
        source_code (str): The Python source code to process.

    Returns:
        str: The code with all comments and docstrings removed.
    """
    # Remove single-line docstrings (triple quotes on same line)
    source_code = re.sub(r'""".*?"""', '', source_code, flags=re.DOTALL)
    source_code = re.sub(r"'''.*?'''", '', source_code, flags=re.DOTALL)

    # Remove multi-line docstrings
    source_code = re.sub(r'"""[\s\S]*?"""', '', source_code)
    source_code = re.sub(r"'''[\s\S]*?'''", '', source_code)

    # Remove comments
    source_code = re.sub(r'#.*$', '', source_code, flags=re.MULTILINE)

    # Clean up extra blank lines
    source_code = re.sub(r'\n\s*\n', '\n\n', source_code)

    return source_code

def mask_strategy_names(source_code):
    """
    Thoroughly mask strategy names throughout the code.
    """
    source_code = strip_comments_and_docstrings(source_code)
    
    class_pattern = r'class\s+(\w+)\s*(?:\(\s*(\w+)\s*\))?:'
    class_matches = re.finditer(class_pattern, source_code)
    
    class_info = {}
    class_order = []
    
    for match in class_matches:
        class_name = match.group(1)
        class_order.append(class_name)
        class_info[class_name] = {
            'base': match.group(2) if match.group(2) else None,
            'masked_name': None
        }
    
    # Create masked names
    replacements = {}
    base_class_counter = 1
    subclass_counter = 1
    
    for class_name in class_order:
        base_class = class_info[class_name]['base']
        
        if base_class is None or base_class not in class_info:
            masked_name = f"BaseStrategy{base_class_counter}"
            base_class_counter += 1
        else:
            masked_name = f"SubStrategy{subclass_counter}"
            subclass_counter += 1
        
        class_info[class_name]['masked_name'] = masked_name
        replacements[class_name] = masked_name
    
    # Replace class names - first in definitions
    for original_name, masked_name in replacements.items():
        # Replace class definition
        pattern = r'class\s+' + re.escape(original_name) + r'\b'
        source_code = re.sub(pattern, f'class {masked_name}', source_code)
    
    # Replace self.name = "OriginalName" with self.name = "MaskedName"
    for original_name, masked_name in replacements.items():
        # Pattern to find self.name or name assignments with the class name
        name_pattern = r'(self\.name\s*=\s*[\'"])[^\'"]*([\'"](.*?))'
        
        # Replace with masked name, preserving the rest of the string
        source_code = re.sub(name_pattern, fr'\1{masked_name}\2', source_code)
        
        # Also handle name = "OriginalName" pattern
        name_pattern = r'(name\s*=\s*[\'"])[^\'"]*([\'"](.*?))'
        source_code = re.sub(name_pattern, fr'\1{masked_name}\2', source_code)
    
    # Replace all other instances of class names
    for original_name, masked_name in replacements.items():
        # Use word boundaries to ensure we only replace whole words
        pattern = r'\b' + re.escape(original_name) + r'\b'
        source_code = re.sub(pattern, masked_name, source_code)
    
    for original_name in replacements.keys():
        if re.search(r'\b' + re.escape(original_name) + r'\b', source_code):
            print(f"WARNING: Found remaining instances of '{original_name}' after masking")
    
    return source_code


def clean_syntax_errors(source_code):
    open_count = source_code.count('(')
    close_count = source_code.count(')')
    if open_count > close_count:
        source_code += ')' * (open_count - close_count)

    # Balance square brackets
    open_count = source_code.count('[')
    close_count = source_code.count(']')
    if open_count > close_count:
        source_code += ']' * (open_count - close_count)

    # Balance curly braces
    open_count = source_code.count('{')
    close_count = source_code.count('}')
    if open_count > close_count:
        source_code += '}' * (open_count - close_count)

    # Fix indentation issues (simplistic approach)
    lines = source_code.split('\n')
    fixed_lines = []

    for line in lines:
        # Strip trailing whitespace
        line = line.rstrip()

        # If line is empty, skip indentation fixing
        if not line.strip():
            fixed_lines.append(line)
            continue

        # Get the indentation level (number of spaces at the beginning)
        indent_match = re.match(r'^(\s*)', line)
        indent = indent_match.group(1) if indent_match else ''

        # Ensure indentation is a multiple of 4 spaces
        if len(indent) % 4 != 0:
            # Round to nearest multiple of 4
            new_indent_level = round(len(indent) / 4)
            new_indent = ' ' * (new_indent_level * 4)
            line = new_indent + line.lstrip()

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def add_program_attributes(refactored_dir="./refactored", obfuscated_dir=None):
    """
    Adds 'program', 'program_masked', and 'program_obfuscated' attributes to
    each strategy in the Axelrod library.

    'program': Stripped of comments and docstrings from refactored_dir.
    'program_masked': Stripped and with strategy names masked.
    'program_obfuscated': Reads pre-obfuscated code from obfuscated_dir.

    Args:
        refactored_dir (str): Directory containing the original strategy files.
        obfuscated_dir (str): Directory containing the pre-obfuscated strategy files
                                (e.g., MyStrategy-obf.py). This is required to set
                                the 'program_obfuscated' attribute.
    """
    if obfuscated_dir is None:
        print("ERROR: obfuscated_dir argument is required to load pre-obfuscated code.")
        sys.exit(1) # Exit if the required directory isn't provided

    strategies = axl.all_strategies
    missing_strategies = []
    obfuscated_missing_count = 0

    print(f"Found {len(strategies)} strategies in Axelrod library")
    print(f"Looking for original program files in: {os.path.abspath(refactored_dir)}")
    print(f"Looking for pre-obfuscated files in: {os.path.abspath(obfuscated_dir)}")


    if not os.path.isdir(refactored_dir):
        print(f"ERROR: Directory not found: {refactored_dir}")
        return
    if not os.path.isdir(obfuscated_dir):
         print(f"ERROR: Directory not found: {obfuscated_dir}")
         return

    processed_count = 0
    obfuscated_loaded_count = 0

    for strategy in strategies:
        strategy_name = strategy.__name__
        strategy_file = os.path.join(refactored_dir, f"{strategy_name}.py")

        if os.path.exists(strategy_file):
            program_text = None
            program_stripped = None
            program_masked = None
            obfuscated_code = None # Initialize as None

            try:
                with open(strategy_file, 'r', encoding='utf-8') as f:
                    program_text = f.read()

                # Try to fix syntax errors before processing
                program_text = clean_syntax_errors(program_text)

                # Create 'program' (stripped only)
                program_stripped = strip_comments_and_docstrings(program_text)
                setattr(strategy, 'program', program_stripped)

                # Create 'program_masked' (stripped and masked)
                # Use the already stripped version for efficiency
                program_masked = mask_strategy_names(program_stripped)
                setattr(strategy, 'program_masked', program_masked)

                processed_count += 1 # Count processing success for original file

                # --- Load pre-obfuscated code ---
                # Construct the expected obfuscated filename (e.g., MyStrategy-obf.py)
                base_name = os.path.basename(strategy_file)[:-3] # Get "MyStrategy"
                obfuscated_filename = f"{base_name}-obf.py"
                obfuscated_file_path = os.path.join(obfuscated_dir, obfuscated_filename)

                if os.path.exists(obfuscated_file_path):
                    try:
                        with open(obfuscated_file_path, 'r', encoding='utf-8') as f_obf:
                            obfuscated_code = f_obf.read()
                        obfuscated_loaded_count += 1 # Count successful loads
                    except Exception as e_obf:
                        print(f"  WARNING: Could not read obfuscated file {obfuscated_file_path}: {e_obf}")
                        obfuscated_missing_count += 1
                        # obfuscated_code remains None
                else:
                    # Only print warning if the original file was processed
                    print(f"  WARNING: Pre-obfuscated file not found: {obfuscated_file_path}")
                    obfuscated_missing_count += 1
                    # obfuscated_code remains None

                # Set the attribute (will be None if file not found or failed to read)
                setattr(strategy, 'program_obfuscated', obfuscated_code)
                # --- End loading pre-obfuscated code ---

            except Exception as e:
                print(f"ERROR: Could not process original file {strategy_file}: {e}")
                missing_strategies.append(strategy_name)
                # Ensure attributes are None if processing failed mid-way
                if not hasattr(strategy, 'program'): setattr(strategy, 'program', None)
                if not hasattr(strategy, 'program_masked'): setattr(strategy, 'program_masked', None)
                if not hasattr(strategy, 'program_obfuscated'): setattr(strategy, 'program_obfuscated', None)
                continue  # Skip to the next strategy
        else:
            missing_strategies.append(strategy_name)

    # --- Updated Summary ---
    print(f"\nFinished processing.")
    print(f"Successfully processed original files for {processed_count} strategies.")
    print(f"Successfully loaded pre-obfuscated code for {obfuscated_loaded_count} strategies.")
    print(f" - 'program': Original code stripped of comments/docstrings.")
    print(f" - 'program_masked': Stripped code with strategy names masked.")
    print(f" - 'program_obfuscated': Loaded from corresponding file in '{os.path.abspath(obfuscated_dir)}'.")

    if missing_strategies:
        print(f"\nWARNING: Could not find or process original program files for {len(missing_strategies)} strategies:")
        for name in missing_strategies[:10]:  # Show only first 10 to avoid long output
            print(f" - {name}")
        if len(missing_strategies) > 10:
            print(f" ...and {len(missing_strategies) - 10} more")

    if obfuscated_missing_count > 0:
         print(f"\nWARNING: Failed to load or find pre-obfuscated files for {obfuscated_missing_count} strategies (set to None).")


    # --- Updated Example Print ---
    if processed_count > 0:
        sample_strategy = next((s for s in strategies if hasattr(s, 'program') and s.program is not None), None)

        if sample_strategy:
            print(f"\nExample - Attributes for {sample_strategy.__name__}:")

            print(f"\n'program' (first 200 chars):")
            program_preview = sample_strategy.program[:200] + "..." if sample_strategy.program and len(sample_strategy.program) > 200 else sample_strategy.program
            print(f"{program_preview if program_preview else '(Not available)'}")

            print(f"\n'program_masked' (first 200 chars):")
            program_masked_preview = sample_strategy.program_masked[:200] + "..." if sample_strategy.program_masked and len(sample_strategy.program_masked) > 200 else sample_strategy.program_masked
            print(f"{program_masked_preview if program_masked_preview else '(Not available)'}")

            print(f"\n'program_obfuscated' (first 200 chars):")
            if hasattr(sample_strategy, 'program_obfuscated') and sample_strategy.program_obfuscated:
                program_obfuscated_preview = sample_strategy.program_obfuscated[:200] + "..." if len(sample_strategy.program_obfuscated) > 200 else sample_strategy.program_obfuscated
                print(f"{program_obfuscated_preview}")
            else:
                print("(Not available - check warnings above)")

            print("\nYou can now access the modified program code using: strategy.program, strategy.program_masked, and strategy.program_obfuscated")

# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Add program attributes to Axelrod strategies by reading original and pre-obfuscated files."
        )

    parser.add_argument("--refactored", default="./refactored",
                      help="Directory containing original strategy files (default: ./refactored)")

    # Made obfuscated directory required
    parser.add_argument("--obfuscated", required=True,
                      help="Directory containing pre-obfuscated strategy files (e.g., MyStrategy-obf.py). REQUIRED.")

    args = parser.parse_args()

    add_program_attributes(args.refactored, args.obfuscated)

#%%