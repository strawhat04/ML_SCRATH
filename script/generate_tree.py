"""Print the repository's src directory tree, including individual files."""

from pathlib import Path


def print_tree(directory: Path, prefix: str = "") -> None:
    entries = sorted(
        (entry for entry in directory.iterdir() if entry.name != "__pycache__"),
        key=lambda entry: (not entry.is_dir(), entry.name.casefold(), entry.name),
    )

    for index, entry in enumerate(entries):
        is_last = index == len(entries) - 1
        branch = "`-- " if is_last else "|-- "
        is_directory = entry.is_dir()
        suffix = "/" if is_directory else ""
        print(f"{prefix}{branch}{entry.name}{suffix}")

        if is_directory and not entry.is_symlink() and not entry.is_junction():
            continuation = "    " if is_last else "|   "
            print_tree(entry, prefix + continuation)


if __name__ == "__main__":
    source = Path(__file__).resolve().parents[1] / "src"
    print(f"{source.name}/")
    print_tree(source)
