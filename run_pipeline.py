"""Run the data pipeline, or the full app with --end-to-end."""

import sys

if __name__ == "__main__":
    if "--end-to-end" in sys.argv:
        from src.pipeline import main
    else:
        from src.data.pipeline import main
    main()
