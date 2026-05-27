import sys
from cs336_alignment.modal_utils import app, quote_command, submit_commands, VOLUME_MOUNTS


def build_run_commands():
    return [
        ["python", "-u", "cs336_alignment/benchmark_olmo.py"]
    ]

@app.local_entrypoint(name="benchmark")
def modal_main(*argv: str) -> None:
    commands = build_run_commands()
    submit_commands(commands)