import logging
import re
import zipfile
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

logger = logging.getLogger(__name__)

INPUT_CMD_RE = re.compile(r"\\(?:input|include|subfile)\b\s*\{([^}]+)\}")


def flatten_tex_zip(zip_bytes: bytes) -> tuple[str, list[str], str]:
    """Extract zip, find main .tex, resolve \\input/\\include/\\subfile recursively.

    Returns (flattened_tex, included_files_in_visit_order, main_file_relpath).
    """
    logger.info("Flattening zip: %d bytes", len(zip_bytes))
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _safe_extract(zip_bytes, root)
        return _flatten_dir(root)


def _safe_extract(zip_bytes: bytes, root: Path) -> None:
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        for name in names:
            target = (root / name).resolve()
            try:
                target.relative_to(root.resolve())
            except ValueError:
                raise ValueError(f"Unsafe zip entry escapes root: {name}")
        zf.extractall(root)
        logger.info("Extracted %d zip entries to %s", len(names), root)


def _flatten_dir(root: Path) -> tuple[str, list[str], str]:
    root = root.resolve()
    tex_files = [f for f in root.rglob("*.tex") if "__MACOSX" not in f.parts]
    logger.info("Found %d .tex file(s) in archive", len(tex_files))
    if not tex_files:
        raise ValueError("Zip contains no .tex files.")

    main_file = _find_main(tex_files)
    logger.info("Detected main file: %s", main_file.relative_to(root))

    included: list[str] = []
    flattened = _resolve(main_file, root, visited=set(), included=included)
    logger.info(
        "Flattened %d file(s), %d total chars",
        len(included), len(flattened),
    )
    return flattened, included, str(main_file.relative_to(root))


def _find_main(tex_files: list[Path]) -> Path:
    candidates: list[Path] = []
    for f in tex_files:
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if r"\documentclass" in content:
            candidates.append(f)
    logger.debug("documentclass candidates: %s", [str(c) for c in candidates])
    if not candidates:
        raise ValueError(
            "No .tex file contains \\documentclass; cannot identify the main file."
        )
    if len(candidates) == 1:
        return candidates[0]
    for preferred in ("main.tex", "resume.tex", "cv.tex"):
        for c in candidates:
            if c.name.lower() == preferred:
                logger.info("Main file picked by preferred-name match: %s", c.name)
                return c
    chosen = min(candidates, key=lambda p: len(p.parts))
    logger.info("Multiple candidates; chose shortest path: %s", chosen)
    return chosen


def _resolve(
    file: Path, root: Path, visited: set[Path], included: list[str]
) -> str:
    real = file.resolve()
    if real in visited:
        raise ValueError(
            f"Circular \\input chain at {file.relative_to(root)}"
        )
    visited.add(real)

    rel = str(file.relative_to(root))
    if rel not in included:
        included.append(rel)
    logger.debug("Resolving file: %s", rel)

    content = file.read_text(encoding="utf-8", errors="replace")

    def replace(match: re.Match) -> str:
        ref = match.group(1).strip()
        target = _find_referenced(ref, file.parent, root)
        if target is None:
            logger.warning("Unresolved reference: \\input{%s} in %s", ref, rel)
            return match.group(0)
        rel_t = target.relative_to(root)
        logger.info("Inlined \\input{%s} -> %s", ref, rel_t)
        inner = _resolve(target, root, visited, included)
        return f"% --- begin {rel_t} ---\n{inner}\n% --- end {rel_t} ---"

    return _sub_skipping_comments(INPUT_CMD_RE, replace, content)


def _sub_skipping_comments(pattern: re.Pattern, repl, text: str) -> str:
    out: list[str] = []
    last = 0
    for m in pattern.finditer(text):
        line_start = text.rfind("\n", 0, m.start()) + 1
        prefix = text[line_start : m.start()]
        if _has_unescaped_percent(prefix):
            continue
        out.append(text[last : m.start()])
        out.append(repl(m))
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def _has_unescaped_percent(prefix: str) -> bool:
    i = 0
    while i < len(prefix):
        if prefix[i] == "\\":
            i += 2
            continue
        if prefix[i] == "%":
            return True
        i += 1
    return False


def _find_referenced(ref: str, current_dir: Path, root: Path) -> Path | None:
    paths_to_try = [ref] if ref.endswith(".tex") else [ref + ".tex", ref]
    root_real = root.resolve()
    for p in paths_to_try:
        for base in (current_dir, root):
            cand = (base / p).resolve()
            try:
                cand.relative_to(root_real)
            except ValueError:
                continue
            if cand.is_file():
                return cand
    return None
