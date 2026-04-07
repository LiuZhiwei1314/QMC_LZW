"""Compatibility launcher.

Use main.py as the canonical project entrypoint.
"""

try:
	from main import main
except ModuleNotFoundError:
	import os
	import sys

	project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	if project_root not in sys.path:
		sys.path.insert(0, project_root)
	from main import main


if __name__ == "__main__":
	main()

