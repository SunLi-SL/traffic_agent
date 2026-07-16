import subprocess
import sys
from pathlib import Path

# 和上面主文件名字一致
app_file = Path(__file__).parent / "map_agent.py"

if __name__ == "__main__":
    subprocess.run(
        [
            sys.executable,
            "-m", "streamlit",
            "run", str(app_file),
            "--server.port=8501"
        ],
        check=True
    )