from os.path import join


def get_path(*arg: str) -> str:
    return join(*arg)
