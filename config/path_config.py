from pathlib import Path

def get_paths() -> Path:
    # Retrieve the paths for the 
    base_path = Path(__file__).parent.parent
    input_path = base_path / "input"
    config_path = base_path / "config"
    code_path = base_path / "code"
    output_path = base_path / "output"

    return input_path, config_path, code_path, output_path
