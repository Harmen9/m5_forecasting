from pathlib import Path

class Paths:
    # The paths for the script. 
    BASE = Path().cwd()
    INPUT = BASE / "input"
    CONFIG = BASE / "code" / "config"
    CODE = BASE / "code"
    OUTPUT = BASE / "output"