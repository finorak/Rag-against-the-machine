"""Program entry point module."""


import sys

from fire import Fire

from .app import App


def main() -> None:
    """Program entry point function."""
    app = App()
    Fire(app)


if __name__ == "__main__":
    try:
        main()
    except ConnectionError as e:
        print(e, file=sys.stderr)
    except Exception as e:
        print(e, file=sys.stderr)
