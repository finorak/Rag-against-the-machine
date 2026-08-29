from fire import Fire

from .app import App


def main() -> None:
    app = App()
    Fire(app)


if __name__ == "__main__":
    main()
