"""Program entry point module."""


from fire import Fire

from .app import App


def main() -> None:
    """Program entry point function."""
    app = App()
    Fire(app)


if __name__ == "__main__":
    main()
