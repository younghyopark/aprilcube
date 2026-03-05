"""CLI entry point for aprilcube."""

import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: aprilcube <command> [args...]\n")
        print("Commands:")
        print("  generate    Generate a 3MF cube with ArUco/AprilTag markers")
        print("\nRun 'aprilcube <command> --help' for command-specific help.")
        sys.exit(0)

    if sys.argv[1] == "--version":
        from aprilcube import __version__
        print(f"aprilcube {__version__}")
        sys.exit(0)

    command = sys.argv[1]
    sys.argv = [f"aprilcube {command}"] + sys.argv[2:]

    if command == "generate":
        from aprilcube.generate import main as generate_main
        generate_main()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Available commands: generate", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
