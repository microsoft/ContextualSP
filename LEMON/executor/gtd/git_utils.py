import git

def commit_diff(c):
    """Return the set of changed files.

    Args:
        c (git.Commit)

    Returns:
        set[str]: a set of file paths (relative to the git repo's root directory).
    """
    changed = set()

    def add_path(blob):
        if blob is not None:
            changed.add(blob.path)

    prev_c = c.parents[0]
    for x in c.diff(prev_c):
        add_path(x.a_blob)
        add_path(x.b_blob)
    return changed