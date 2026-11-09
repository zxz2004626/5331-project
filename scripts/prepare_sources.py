"""Fetch only the pinned public upstream, without overwriting an existing checkout."""
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
REV="af9d575e07c7bc3fd0b4a0b8d0249e7cdd2eb88b"
TARGET=ROOT/"vendor/TGF"

def main():
    if not TARGET.exists():
        TARGET.parent.mkdir(exist_ok=True)
        subprocess.run(["git","clone","https://github.com/StephanyRajeh/TGF.git",str(TARGET)],check=True)
        subprocess.run(["git","-C",str(TARGET),"checkout","--detach",REV],check=True)
    rev=subprocess.check_output(["git","-C",str(TARGET),"rev-parse","HEAD"],text=True).strip()
    if rev!=REV:raise RuntimeError(f"Existing upstream is {rev}, expected {REV}. Preserve its changes and select the pinned revision explicitly.")
    print(f"Verified upstream {rev}")

if __name__=="__main__":main()
