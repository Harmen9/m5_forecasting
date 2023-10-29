from pathlib import Path
from dataclasses import (
    dataclass,
)

@dataclass
class Paths:
    'Contains the folder structure for the M5 project'
    base: Path
    input: Path
    config: Path
    code: Path
    output: Path

    def __init__(
        self,
        input_path: str,
        config_path: str,
        code_path: str,
        output_path: str
    ) -> None:
        self.base = Path().cwd()
        self.input = Path().cwd() / input_path
        self.config = Path().cwd() / config_path
        self.code = Path().cwd() / code_path
        self.output = Path().cwd() / output_path

