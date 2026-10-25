"""Generate the docs/examples/ pages from the notebooks in examples/<category>/*.ipynb.

All output is build-only and is not committed (docs/examples/ is in .gitignore):

1. docs/examples/index.md: overview listing the categories, then every example.
2. docs/examples/<category>.md: card gallery of the examples in one category.
3. docs/examples/<stem>.md: one page per notebook, keras.io/examples style, with
   'View in Colab' and 'GitHub source' buttons, the notebook's markdown and code
   cells, and any saved outputs.

The category of an example is its parent directory under examples/, and its title and
description come from the notebook's first markdown cell, so adding or moving a
notebook is all that's needed to update the docs.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "examples"
DOCS_DIR = ROOT / "docs" / "examples"

REPO_URL = "https://github.com/anas-rz/keras-sciml/blob/main"
COLAB_URL = "https://colab.research.google.com/github/anas-rz/keras-sciml/blob/main"

# Display metadata for each examples/<category>/ directory, in display order.
CATEGORY_META = {
    "neural_operators": {
        "title": "Neural Operators",
        "icon": ":material-function-variant:",
        "description": "Learn solution operators of PDEs from data with FNO and DeepONet.",
    },
    "physics_informed": {
        "title": "Physics-Informed Learning",
        "icon": ":material-atom:",
        "description": "Train networks and operators against PDE residuals and boundary conditions.",
    },
    "climate": {
        "title": "Climate & Earth Science",
        "icon": ":material-earth:",
        "description": "Neural operators applied to real-world geophysical data.",
    },
}


def category_meta(category):
    return CATEGORY_META.get(category, {
        "title": category.replace("_", " ").title(),
        "icon": ":material-folder-outline:",
        "description": "",
    })


def _imported(code, module):
    names = []
    for line in re.findall(rf"from keras_sciml\.{module} import ([^\n]+)", code):
        names += [n.strip() for n in line.split(",") if n.strip()]
    return names


def get_example_meta(nb_path):
    cells = json.loads(nb_path.read_text(encoding="utf-8"))["cells"]
    markdown = ["".join(c["source"]) for c in cells if c["cell_type"] == "markdown"]
    code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")

    stem = nb_path.stem
    title, description = stem.replace("_", " ").title(), ""
    if markdown and markdown[0].startswith("# "):
        head, _, rest = markdown[0].partition("\n")
        title = head[2:].strip()
        intro = " ".join(rest.strip().split("\n\n")[0].split())
        intro = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", intro)  # markdown links -> text
        description = re.split(r"(?<![ie]\.[eg]\.)(?<=[.!?])\s", intro, maxsplit=1)[0]
        description = re.sub(r":$", ".", description)  # sentence leading into an equation

    models = [n for n in _imported(code, "models") if n[:1].isupper()]
    datasets = _imported(code, "datasets")
    dataset = datasets[0] if datasets else "Synthetic"
    if dataset.startswith("load_"):
        dataset = dataset[len("load_"):].replace("_", " ").title()

    rel_path = nb_path.relative_to(ROOT).as_posix()
    return {
        "title": title,
        "description": description,
        "model": " · ".join(models) or "keras-sciml",
        "dataset": dataset,
        "stem": stem,
        "category": nb_path.parent.name,
        "doc_file": f"{stem}.md",
        "github_url": f"{REPO_URL}/{rel_path}",
        "colab_url": f"{COLAB_URL}/{rel_path}",
    }


def parse_cell_output(cell):
    chunks = []
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            chunks.append("".join(out.get("text", [])))
        elif out.get("output_type") in ("execute_result", "display_data"):
            data = out.get("data", {})
            if "text/plain" in data:
                chunks.append("".join(data["text/plain"]))
    return "\n".join(chunks).strip()


def generate_single_example_page(nb_path, meta):
    cells = json.loads(nb_path.read_text(encoding="utf-8"))["cells"]

    lines = [
        f"# {meta['title']}",
        "",
        "**Author:** keras-sciml Team<br>",
        "**Backend:** Multi-Backend<br>",
        f"**Model:** `{meta['model']}`<br>",
        f"**Dataset:** `{meta['dataset']}`<br>",
        f"**Description:** {meta['description']}",
        "",
        f'[:simple-googlecolab: **View in Colab**]({meta["colab_url"]}){{ .md-button .md-button--primary }} &nbsp; '
        f'[:octicons-mark-github-16: **GitHub source**]({meta["github_url"]}){{ .md-button }}',
        "",
        "---",
        "",
    ]

    for idx, cell in enumerate(cells):
        src = "".join(cell.get("source", [])).strip()
        if not src:
            continue

        if cell["cell_type"] == "markdown":
            # The title is already rendered above
            if idx == 0 and src.startswith("# "):
                src = src.partition("\n")[2].strip()
            lines += [src, ""]
            continue

        src_lines = src.splitlines()
        shell = [l[1:] for l in src_lines if l.startswith("!")]
        if shell:
            # Install/backend cell: shell commands as bash, the rest as python
            python = [l for l in src_lines if not l.startswith("!") and not l.startswith("# ")]
            lines += ["## Setup", "", "```bash", *shell, "```", ""]
            if python:
                lines += ["```python", *python, "```", ""]
            continue

        lines += ["```python", src, "```", ""]

        output = parse_cell_output(cell)
        if output:
            lines.append('??? example "View Output"')
            lines.append("    ```text")
            lines += [f"    {l}" for l in output.splitlines()]
            lines += ["    ```", ""]

    return "\n".join(lines)


def generate_category_page(category, examples):
    meta = category_meta(category)
    lines = [
        f"# {meta['title']}",
        "",
        meta["description"],
        "",
        "[:octicons-arrow-left-24: All categories](index.md)",
        "",
        "---",
        "",
        '<div class="grid cards" markdown>',
        "",
    ]
    for ex in sorted(examples, key=lambda e: e["title"]):
        lines += [
            f"-   {meta['icon']} __{ex['title']}__",
            "",
            f"    {ex['description']}",
            "",
            f"    `{ex['dataset']}` · `{ex['model']}`",
            "",
            f"    [:octicons-arrow-right-24: Read Tutorial]({ex['doc_file']}){{ .md-button .md-button--primary }} &nbsp; "
            f"[:simple-googlecolab:]({ex['colab_url']}){{ .md-button title=\"View in Colab\" }} &nbsp; "
            f"[:octicons-mark-github-16:]({ex['github_url']}){{ .md-button title=\"GitHub source\" }}",
            "",
        ]
    lines += ["</div>", ""]
    return "\n".join(lines)


def generate_index_page(examples_by_category):
    order = list(CATEGORY_META)
    categories = sorted(
        examples_by_category,
        key=lambda c: (order.index(c) if c in order else len(order), c),
    )

    lines = [
        "# Code Examples",
        "",
        "Welcome to the **keras-sciml Code Examples**. Every example is a notebook that runs "
        "on TensorFlow, PyTorch or JAX. Browse by category, or jump straight to a notebook below.",
        "",
        "## Categories",
        "",
    ]
    for category in categories:
        count = len(examples_by_category[category])
        lines.append(
            f"- [{category_meta(category)['title']}]({category}.md) — "
            f"{count} example{'s' if count != 1 else ''}"
        )

    lines += ["", "## All Examples", ""]
    all_examples = sorted(
        (ex for examples in examples_by_category.values() for ex in examples),
        key=lambda e: e["title"],
    )
    for ex in all_examples:
        lines.append(f"- [{ex['title']}]({ex['doc_file']}) — *{category_meta(ex['category'])['title']}*")
    lines.append("")
    return "\n".join(lines)


def _write(path, content):
    # Only touch changed files so `mkdocs serve` doesn't rebuild in a loop
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")


def generate_all():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    examples_by_category = {}
    for nb_path in sorted(EXAMPLES_DIR.glob("*/*.ipynb")):
        meta = get_example_meta(nb_path)
        examples_by_category.setdefault(meta["category"], []).append(meta)
        _write(DOCS_DIR / meta["doc_file"], generate_single_example_page(nb_path, meta))

    for category, examples in examples_by_category.items():
        _write(DOCS_DIR / f"{category}.md", generate_category_page(category, examples))

    _write(DOCS_DIR / "index.md", generate_index_page(examples_by_category))


if __name__ == "__main__":
    generate_all()
