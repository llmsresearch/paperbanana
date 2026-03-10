"""Standalone launcher for PaperBanana Gradio app.

Usage:
    python run_app.py
    python run_app.py --port 8080
    python run_app.py --share
"""

import argparse
import sys

# Ensure UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(description="Launch PaperBanana web application")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    from paperbanana.app.main import create_app

    print(f"Launching PaperBanana app on http://localhost:{args.port}")
    demo = create_app()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
