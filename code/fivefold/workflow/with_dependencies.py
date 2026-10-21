"""Run a companion analysis script in the experiment's local environment."""
from pathlib import Path
import runpy,sys
code=Path(__file__).resolve().parent
script=(code/sys.argv[1]).resolve()
if script.parent!=code or script.suffix!='.py':
    raise SystemExit('Specify a Python companion script in this code directory.')
if script.parent != code or script.suffix != '.py':
    raise ValueError('Choose a Python companion in this workflow directory.')
sys.argv=sys.argv[1:]
runpy.run_path(str(script),run_name='__main__')
