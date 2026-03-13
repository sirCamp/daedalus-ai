"""Daedalus CLI — command-line interface for AI research assistant."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import click
import yaml
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .core.config import ExperimentConfig, config_diff
from .core.experiment import Experiment, ExperimentStatus, Reflection
from .core.hypothesis import Hypothesis, Prediction
from .core.ledger import Ledger
from .agent.context import ContextBuilder
from .literature.library import Library
from .literature.paper import Paper
from .literature.search import PaperSearcher

console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_project(ctx: click.Context) -> Path:
    """Find the project root by looking for daedalus.yaml."""
    start = Path(ctx.obj or ".").resolve()
    for parent in [start] + list(start.parents):
        if (parent / "daedalus.yaml").exists():
            return parent
    # Fallback to cwd
    return Path.cwd()


def _load_project_config(project_path: Path) -> dict:
    """Load daedalus.yaml."""
    config_file = project_path / "daedalus.yaml"
    if config_file.exists():
        return yaml.safe_load(config_file.read_text()) or {}
    return {}


# ---------------------------------------------------------------------------
# CLI Group
# ---------------------------------------------------------------------------

@click.group()
@click.option("--project", "-p", type=click.Path(exists=True), default=None,
              help="Project directory (default: auto-detect)")
@click.pass_context
def cli(ctx: click.Context, project: Optional[str]) -> None:
    """Daedalus — AI research assistant for hypothesis-driven ML experiments."""
    ctx.ensure_object(dict)
    ctx.obj = project


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("name")
@click.option("--path", type=click.Path(), default=".",
              help="Parent directory (default: current)")
def init(name: str, path: str) -> None:
    """Initialize a new research project."""
    project_dir = Path(path) / name
    if project_dir.exists():
        console.print(f"[red]Directory {project_dir} already exists.[/red]")
        raise SystemExit(1)

    # Scaffold
    project_dir.mkdir(parents=True)
    (project_dir / "ledger").mkdir()
    (project_dir / "ledger" / "experiments.jsonl").touch()
    (project_dir / "ledger" / "papers.jsonl").touch()

    # program.md
    (project_dir / "program.md").write_text(
        f"# {name} — Research Program\n\n"
        "## Goals\n\n"
        "Describe your research goals here.\n\n"
        "## Metrics\n\n"
        "Define your evaluation metrics and what 'better' means.\n\n"
        "## Constraints\n\n"
        "Budget, hardware, timeline, etc.\n\n"
        "## Stack\n\n"
        "Libraries and frameworks for this project:\n\n"
        "- **Training**: (e.g., transformers, trl, pytorch)\n"
        "- **Data**: (e.g., datasets, pandas)\n"
        "- **Evaluation**: (e.g., evaluate, scikit-learn)\n"
        "- **Serving**: (e.g., vllm, text-generation-inference)\n\n"
        "Reference docs:\n"
        "- (add links to relevant docs here)\n"
    )

    # daedalus.yaml
    (project_dir / "daedalus.yaml").write_text(
        f"project_name: {name}\n"
        "\n"
        "runner:\n"
        "  type: local\n"
        "\n"
        "higher_is_better: {}\n"
        "  # accuracy: true\n"
        "  # loss: false\n"
        "\n"
        "# Stack — libraries and frameworks. The agent uses this to write correct code.\n"
        "stack:\n"
        "  python: \"3.11\"\n"
        "  requirements: requirements.txt    # or pyproject.toml, environment.yml\n"
        "  libraries:\n"
        "    - transformers\n"
        "    - datasets\n"
        "    - torch\n"
        "  docs:\n"
        "    - https://huggingface.co/docs/transformers/\n"
        "  # env_setup: |                    # commands to create env on remote\n"
        "  #   conda create -n myproject python=3.11 -y\n"
        "  #   conda activate myproject\n"
        "  #   pip install -r requirements.txt\n"
        "\n"
        "# Scripts registry — tells the agent what scripts are available\n"
        "# and what parameters they accept.\n"
        "scripts:\n"
        "  train:\n"
        "    path: train.py\n"
        "    description: \"Main training script\"\n"
        "    parameters:\n"
        "      model_name:\n"
        "        type: str\n"
        "        default: \"bert-base-uncased\"\n"
        "        description: \"Model to fine-tune\"\n"
        "      learning_rate:\n"
        "        type: float\n"
        "        default: 2e-5\n"
        "        range: [1e-6, 1e-3]\n"
        "      epochs:\n"
        "        type: int\n"
        "        default: 3\n"
        "  # eval:\n"
        "  #   path: evaluate.py\n"
        "  #   description: \"Run evaluation\"\n"
        "  #   parameters:\n"
        "  #     model_path:\n"
        "  #       type: str\n"
        "  #       required: true\n"
    )

    # runner_config.yaml template
    (project_dir / "runner_config.yaml").write_text(
        "# Local runner config\n"
        "local:\n"
        "  python: python\n"
        "\n"
        "# Multi-host SSH (preferred)\n"
        "hosts:\n"
        "  gpu-1:\n"
        "    host: your-gpu-host\n"
        "    user: ubuntu\n"
        "    key_path: ~/.ssh/id_rsa\n"
        "    remote_work_dir: /tmp/daedalus\n"
        "    python_path: python\n"
        "  # gpu-2:\n"
        "  #   host: another-host\n"
        "  #   user: ubuntu\n"
        "  #   ...\n"
    )

    console.print(f"[green]Project '{name}' initialized at {project_dir}[/green]")
    console.print(f"  Edit [bold]{project_dir}/program.md[/bold] to define your research goals.")
    console.print(f"  Edit [bold]{project_dir}/runner_config.yaml[/bold] to configure your runner.")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show current project status."""
    project = _find_project(ctx)
    ledger = Ledger(project / "ledger")

    experiments = ledger.all()
    if not experiments:
        console.print("[dim]No experiments yet. Use 'daedalus hypothesis' to get started.[/dim]")
        return

    # Status counts
    counts: dict[str, int] = {}
    for exp in experiments:
        counts[exp.status.value] = counts.get(exp.status.value, 0) + 1

    table = Table(title="Project Status")
    table.add_column("Status", style="bold")
    table.add_column("Count", justify="right")

    status_styles = {
        "draft": "dim", "queued": "yellow", "running": "blue",
        "completed": "green", "analyzed": "cyan",
        "failed": "red", "abandoned": "dim red",
    }
    for s, count in sorted(counts.items()):
        table.add_row(f"[{status_styles.get(s, '')}]{s}[/]", str(count))
    console.print(table)

    # Active experiments
    active = [e for e in experiments if e.status in (ExperimentStatus.RUNNING, ExperimentStatus.QUEUED)]
    if active:
        console.print("\n[bold]Active experiments:[/bold]")
        for exp in active:
            console.print(f"  🔄 {exp.id}: {exp.hypothesis.statement}")

    # Latest completed
    completed = [e for e in experiments if e.status in (ExperimentStatus.COMPLETED, ExperimentStatus.ANALYZED)]
    if completed:
        latest = completed[-1]
        console.print(f"\n[bold]Latest result:[/bold] {latest.id}")
        console.print(f"  Hypothesis: {latest.hypothesis.statement}")
        if latest.results:
            for eval_name, metrics in latest.results.items():
                metrics_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
                console.print(f"  {eval_name}: {metrics_str}")


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--last", "-n", type=int, default=10, help="Show last N experiments")
@click.pass_context
def ledger(ctx: click.Context, last: int) -> None:
    """Show experiment history."""
    project = _find_project(ctx)
    led = Ledger(project / "ledger")

    experiments = led.latest(last)
    if not experiments:
        console.print("[dim]No experiments yet.[/dim]")
        return

    table = Table(title=f"Last {last} Experiments")
    table.add_column("ID", style="bold")
    table.add_column("Status")
    table.add_column("Hypothesis")
    table.add_column("Result", justify="right")
    table.add_column("Date")

    for exp in reversed(experiments):  # chronological
        result_str = ""
        if exp.reflection:
            result_str = exp.reflection.hypothesis_confirmed

        status_styles = {
            "draft": "dim", "queued": "yellow", "running": "blue",
            "completed": "green", "analyzed": "cyan",
            "failed": "red", "abandoned": "dim red",
        }
        style = status_styles.get(exp.status.value, "")

        table.add_row(
            exp.id,
            f"[{style}]{exp.status.value}[/]",
            exp.hypothesis.statement[:60],
            result_str,
            exp.created_at.strftime("%Y-%m-%d"),
        )

    console.print(table)


# ---------------------------------------------------------------------------
# hypothesis
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("statement")
@click.option("--rationale", "-r", required=True, help="Why you expect this")
@click.option("--predict", "-p", multiple=True,
              help="Prediction: 'metric:direction[:expected_value]'")
@click.option("--paper", multiple=True, help="Supporting arxiv IDs")
@click.option("--baseline", "-b", type=str, default=None, help="Baseline experiment ID")
@click.option("--script", "-s", type=str, default=None, help="Training script path")
@click.option("--arg", "-a", multiple=True, help="Script args: 'key=value'")
@click.pass_context
def hypothesis(
    ctx: click.Context,
    statement: str,
    rationale: str,
    predict: tuple[str, ...],
    paper: tuple[str, ...],
    baseline: Optional[str],
    script: Optional[str],
    arg: tuple[str, ...],
) -> None:
    """Register a new hypothesis as a draft experiment."""
    project = _find_project(ctx)
    led = Ledger(project / "ledger")

    # Parse predictions
    predictions: list[Prediction] = []
    for p in predict:
        parts = p.split(":")
        if len(parts) < 2:
            console.print(f"[red]Invalid prediction format: {p}. Use metric:direction[:value][/red]")
            raise SystemExit(1)
        metric, direction = parts[0], parts[1]
        expected = float(parts[2]) if len(parts) > 2 else None
        predictions.append(Prediction(
            metric=metric,
            direction=direction,  # type: ignore
            expected_value=expected,
        ))

    # Parse script args
    script_args: dict = {}
    for a in arg:
        if "=" not in a:
            console.print(f"[red]Invalid arg format: {a}. Use key=value[/red]")
            raise SystemExit(1)
        k, v = a.split("=", 1)
        # Try to parse as number
        try:
            v = float(v)  # type: ignore
            if v == int(v):
                v = int(v)  # type: ignore
        except ValueError:
            pass
        script_args[k] = v

    # Build experiment
    hyp = Hypothesis(
        statement=statement,
        rationale=rationale,
        predictions=predictions,
        papers=list(paper),
    )

    config = ExperimentConfig(
        script=script or "train.py",
        script_args=script_args,
    )

    # Compute diff from baseline if provided
    diff = None
    if baseline:
        base_exp = led.get(baseline)
        if base_exp:
            diff = config_diff(base_exp.config, config)

    exp = Experiment(
        hypothesis=hyp,
        config=config,
        config_diff=diff,
        baseline_id=baseline,
    )

    led.append(exp)
    console.print(f"[green]Experiment {exp.id} created (draft)[/green]")
    console.print(f"  Hypothesis: {statement}")
    if predictions:
        for pred in predictions:
            console.print(f"  Predict: {pred.metric} → {pred.direction}")


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("exp_a")
@click.argument("exp_b")
@click.pass_context
def diff(ctx: click.Context, exp_a: str, exp_b: str) -> None:
    """Compare two experiments (config + results)."""
    project = _find_project(ctx)
    led = Ledger(project / "ledger")

    a = led.get(exp_a)
    b = led.get(exp_b)

    if not a:
        console.print(f"[red]Experiment {exp_a} not found.[/red]")
        raise SystemExit(1)
    if not b:
        console.print(f"[red]Experiment {exp_b} not found.[/red]")
        raise SystemExit(1)

    # Config diff
    cdiff = config_diff(a.config, b.config)
    if cdiff:
        table = Table(title=f"Config Diff: {exp_a} → {exp_b}")
        table.add_column("Parameter")
        table.add_column(exp_a, justify="right")
        table.add_column(exp_b, justify="right")
        for key, change in cdiff.items():
            table.add_row(key, str(change.get("from", "")), str(change.get("to", "")))
        console.print(table)
    else:
        console.print("[dim]No config differences.[/dim]")

    # Results comparison
    if a.results and b.results:
        from .evaluators.metric import compare_results
        comps = compare_results(b.results, a.results)
        if comps:
            table = Table(title=f"Results: {exp_a} → {exp_b}")
            table.add_column("Metric")
            table.add_column(exp_a, justify="right")
            table.add_column(exp_b, justify="right")
            table.add_column("Delta", justify="right")

            for c in comps:
                delta_style = "green" if c.improved else "red"
                delta_str = f"[{delta_style}]{c.delta:+.3f}[/]"
                table.add_row(
                    c.metric,
                    f"{c.baseline_value:.3f}",
                    f"{c.current_value:.3f}",
                    delta_str,
                )
            console.print(table)


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--mode", "-m", type=click.Choice(["reason", "design", "reflect", "review", "full"]),
              default="full", help="Context mode")
@click.option("--recent", "-n", type=int, default=10, help="Recent experiments to include")
@click.pass_context
def context(ctx: click.Context, mode: str, recent: int) -> None:
    """Dump structured context for agent consumption."""
    project = _find_project(ctx)
    builder = ContextBuilder(project)
    output = builder.build(mode=mode, recent_n=recent)  # type: ignore
    console.print(Markdown(output))


# ---------------------------------------------------------------------------
# narrative
# ---------------------------------------------------------------------------

@cli.command()
@click.pass_context
def narrative(ctx: click.Context) -> None:
    """Show the auto-generated research narrative."""
    project = _find_project(ctx)
    led = Ledger(project / "ledger")
    text = led.narrative()
    if text:
        console.print(Markdown(text))
    else:
        console.print("[dim]No narrative yet. Add experiments first.[/dim]")


# ---------------------------------------------------------------------------
# papers
# ---------------------------------------------------------------------------

@cli.group()
def papers() -> None:
    """Literature management."""
    pass


@papers.command("search")
@click.argument("query")
@click.option("--limit", "-n", type=int, default=5, help="Max results")
@click.pass_context
def papers_search(ctx: click.Context, query: str, limit: int) -> None:
    """Search papers on Semantic Scholar."""
    searcher = PaperSearcher()
    results = searcher.search(query, limit=limit)

    if not results:
        console.print("[dim]No results found.[/dim]")
        return

    table = Table(title=f"Papers: '{query}'")
    table.add_column("#", style="dim", width=3)
    table.add_column("Title")
    table.add_column("Year", justify="right", width=6)
    table.add_column("Citations", justify="right", width=10)
    table.add_column("ArXiv", width=15)

    for i, p in enumerate(results, 1):
        table.add_row(
            str(i),
            p.title[:80],
            str(p.year or ""),
            str(p.citation_count or ""),
            p.arxiv_id or "",
        )

    console.print(table)
    console.print("\n[dim]Use 'daedalus papers add <arxiv_id>' to add to library.[/dim]")


@papers.command("add")
@click.argument("arxiv_id")
@click.option("--note", "-n", type=str, default="", help="Relevance note")
@click.option("--tag", "-t", multiple=True, help="Tags")
@click.pass_context
def papers_add(ctx: click.Context, arxiv_id: str, note: str, tag: tuple[str, ...]) -> None:
    """Add a paper to the library by arxiv ID."""
    project = _find_project(ctx)
    lib = Library(project / "ledger" / "papers.jsonl")

    searcher = PaperSearcher()
    paper = searcher.get_paper(f"ARXIV:{arxiv_id}")

    if paper is None:
        console.print(f"[red]Paper arXiv:{arxiv_id} not found.[/red]")
        raise SystemExit(1)

    paper.relevance_note = note
    paper.tags = list(tag)
    lib.add(paper)

    console.print(f"[green]Added: {paper.title}[/green]")
    if paper.authors:
        console.print(f"  Authors: {', '.join(paper.authors[:3])}")


@papers.command("list")
@click.option("--tag", "-t", type=str, default=None, help="Filter by tag")
@click.pass_context
def papers_list(ctx: click.Context, tag: Optional[str]) -> None:
    """List papers in the library."""
    project = _find_project(ctx)
    lib = Library(project / "ledger" / "papers.jsonl")

    papers_list = lib.by_tag(tag) if tag else lib.all()

    if not papers_list:
        console.print("[dim]Library is empty.[/dim]")
        return

    table = Table(title="Paper Library")
    table.add_column("ArXiv", width=15)
    table.add_column("Title")
    table.add_column("Year", justify="right", width=6)
    table.add_column("Tags")
    table.add_column("Note")

    for p in papers_list:
        table.add_row(
            p.arxiv_id or "",
            p.title[:60],
            str(p.year or ""),
            ", ".join(p.tags),
            (p.relevance_note[:40] + "...") if len(p.relevance_note) > 40 else p.relevance_note,
        )

    console.print(table)


cli.add_command(papers)


# ---------------------------------------------------------------------------
# data (dataset inspection & validation)
# ---------------------------------------------------------------------------

@cli.group()
def data() -> None:
    """Dataset inspection and validation."""
    pass


@data.command("inspect")
@click.argument("path")
@click.option("--sample-n", type=int, default=None,
              help="Sample N records for large datasets")
@click.option("--json-output", is_flag=True, help="Output as JSON")
def data_inspect(path: str, sample_n: Optional[int], json_output: bool) -> None:
    """Inspect a dataset: stats, distributions, sample records."""
    from .data.inspector import DatasetInspector

    inspector = DatasetInspector()
    report = inspector.inspect(path, sample_n=sample_n)

    if json_output:
        console.print_json(report.model_dump_json())
        return

    # Summary
    console.print(Panel(
        f"[bold]Rows:[/bold] {report.row_count}  "
        f"[bold]Columns:[/bold] {report.column_count}  "
        f"[bold]Format:[/bold] {report.format}"
        + (f"  [bold]Size:[/bold] {report.file_size_bytes:,} bytes" if report.file_size_bytes else "")
        + (f"  [bold]Est. tokens:[/bold] {report.estimated_tokens:,}" if report.estimated_tokens else "")
        + (" [yellow](sampled)[/yellow]" if report.sampled else ""),
        title="Dataset Inspection",
    ))

    # Column stats table
    table = Table(title="Columns")
    table.add_column("Field", style="bold")
    table.add_column("Type")
    table.add_column("Nulls", justify="right")
    table.add_column("Unique", justify="right")
    table.add_column("Empty", justify="right")
    table.add_column("Top Values")

    for col in report.columns:
        top = ", ".join(f"{v}({c})" for v, c in col.top_values[:3])
        num_info = ""
        if col.min_value is not None:
            num_info = f"[{col.min_value:.2f}, {col.max_value:.2f}] avg={col.mean_value:.2f}"
            top = num_info if not top else top

        table.add_row(
            col.name,
            col.dtype,
            str(col.null_count),
            str(col.unique_count),
            str(col.empty_string_count),
            top[:60],
        )

    console.print(table)

    # Sample records
    if report.sample_records:
        console.print("\n[bold]Sample Records:[/bold]")
        for i, rec in enumerate(report.sample_records[:3], 1):
            # Truncate long values
            display = {}
            for k, v in rec.items():
                sv = str(v)
                display[k] = sv[:100] + "..." if len(sv) > 100 else sv
            console.print(f"  [{i}] {display}")

    for w in report.warnings:
        console.print(f"  [yellow]Warning: {w}[/yellow]")


@data.command("validate")
@click.argument("path")
@click.option("--format", "-f", "fmt", type=click.Choice(["sft", "dpo", "grpo", "chat"]),
              default=None, help="Expected format")
@click.option("--required-field", "-r", multiple=True, help="Required field name")
@click.option("--max-duplicates", type=float, default=0.01,
              help="Max duplicate ratio (default: 0.01)")
@click.option("--max-empty", type=float, default=0.05,
              help="Max empty field ratio (default: 0.05)")
@click.option("--check-balance", is_flag=True, help="Check class balance")
@click.option("--label-field", type=str, default=None,
              help="Field for class balance check")
@click.option("--eval-path", type=str, default=None,
              help="Eval split path for leakage check")
@click.option("--sample-n", type=int, default=None,
              help="Sample N records")
def data_validate(
    path: str,
    fmt: Optional[str],
    required_field: tuple[str, ...],
    max_duplicates: float,
    max_empty: float,
    check_balance: bool,
    label_field: Optional[str],
    eval_path: Optional[str],
    sample_n: Optional[int],
) -> None:
    """Validate a dataset against configurable checks."""
    from .data.validator import DatasetValidator, ValidatorConfig

    config = ValidatorConfig(
        required_fields=list(required_field) if required_field else None,
        format=fmt,
        max_duplicate_ratio=max_duplicates,
        max_empty_ratio=max_empty,
        check_class_balance=check_balance,
        label_field=label_field,
        eval_path=eval_path,
    )

    validator = DatasetValidator(config)
    report = validator.validate(path, sample_n=sample_n)

    table = Table(title="Validation Results")
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Severity")
    table.add_column("Message")

    for check in report.checks:
        status = "[green]PASS[/green]" if check.passed else "[red]FAIL[/red]"
        sev_style = {"error": "red", "warning": "yellow", "info": "dim"}.get(check.severity, "")
        table.add_row(
            check.name,
            status,
            f"[{sev_style}]{check.severity}[/]",
            check.message[:80],
        )

    console.print(table)

    if report.passed:
        console.print("\n[green]All checks passed.[/green]")
    else:
        console.print(f"\n[red]{report.error_count} error(s), {report.warning_count} warning(s)[/red]")
        raise SystemExit(1)


cli.add_command(data)


# ---------------------------------------------------------------------------
# record (manual results entry)
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("exp_id")
@click.option("--result", "-r", multiple=True,
              help="Result: 'eval_name.metric=value'")
@click.option("--status", "-s", type=click.Choice(["completed", "failed", "abandoned"]),
              default="completed")
@click.pass_context
def record(ctx: click.Context, exp_id: str, result: tuple[str, ...], status: str) -> None:
    """Record results for an experiment."""
    project = _find_project(ctx)
    led = Ledger(project / "ledger")

    exp = led.get(exp_id)
    if not exp:
        console.print(f"[red]Experiment {exp_id} not found.[/red]")
        raise SystemExit(1)

    # Parse results
    results: dict[str, dict[str, float]] = {}
    for r in result:
        if "=" not in r:
            console.print(f"[red]Invalid result format: {r}. Use eval.metric=value[/red]")
            raise SystemExit(1)
        key, val = r.split("=", 1)
        parts = key.split(".", 1)
        if len(parts) != 2:
            console.print(f"[red]Invalid metric key: {key}. Use eval_name.metric[/red]")
            raise SystemExit(1)
        eval_name, metric = parts
        if eval_name not in results:
            results[eval_name] = {}
        results[eval_name][metric] = float(val)

    # Transition status
    target_status = ExperimentStatus(status)
    try:
        # May need intermediate transitions
        if exp.status == ExperimentStatus.DRAFT:
            exp = exp.transition(ExperimentStatus.QUEUED)
        if exp.status == ExperimentStatus.QUEUED:
            exp = exp.transition(ExperimentStatus.RUNNING)
        if exp.status == ExperimentStatus.RUNNING and target_status in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED):
            exp = exp.transition(target_status)
    except ValueError as e:
        console.print(f"[red]Status transition error: {e}[/red]")
        raise SystemExit(1)

    exp = exp.model_copy(update={"results": results if results else exp.results})
    led.update(exp)

    console.print(f"[green]Experiment {exp_id} → {status}[/green]")
    if results:
        for eval_name, metrics in results.items():
            for k, v in metrics.items():
                console.print(f"  {eval_name}.{k} = {v}")


# ---------------------------------------------------------------------------
# reflect (manual reflection entry)
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("exp_id")
@click.option("--analysis", "-a", required=True, help="Analysis text")
@click.option("--confirmed", "-c",
              type=click.Choice(["confirmed", "partial", "rejected"]),
              required=True)
@click.option("--surprise", type=str, default=None)
@click.option("--next", "-n", "next_steps", multiple=True, help="Suggested next steps")
@click.pass_context
def reflect(
    ctx: click.Context,
    exp_id: str,
    analysis: str,
    confirmed: str,
    surprise: Optional[str],
    next_steps: tuple[str, ...],
) -> None:
    """Add reflection to a completed experiment."""
    project = _find_project(ctx)
    led = Ledger(project / "ledger")

    exp = led.get(exp_id)
    if not exp:
        console.print(f"[red]Experiment {exp_id} not found.[/red]")
        raise SystemExit(1)

    reflection = Reflection(
        hypothesis_confirmed=confirmed,
        analysis=analysis,
        surprise=surprise,
        next_suggestions=list(next_steps),
    )

    # Transition to analyzed if completed
    if exp.status == ExperimentStatus.COMPLETED:
        exp = exp.transition(ExperimentStatus.ANALYZED)

    exp = exp.model_copy(update={"reflection": reflection})

    # Also evaluate hypothesis
    if exp.results:
        flat_results = {}
        for eval_name, metrics in exp.results.items():
            for k, v in metrics.items():
                flat_results[f"{eval_name}.{k}"] = v

        flat_baseline = None
        if exp.baseline_id:
            base_exp = led.get(exp.baseline_id)
            if base_exp and base_exp.results:
                flat_baseline = {}
                for eval_name, metrics in base_exp.results.items():
                    for k, v in metrics.items():
                        flat_baseline[f"{eval_name}.{k}"] = v

        evaluated_hyp = exp.hypothesis.evaluate(flat_results, flat_baseline)
        exp = exp.model_copy(update={"hypothesis": evaluated_hyp})

    led.update(exp)

    console.print(f"[green]Reflection added to {exp_id} → analyzed[/green]")
    console.print(f"  Hypothesis: {confirmed}")
    console.print(f"  Analysis: {analysis}")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("exp_id")
@click.option("--confirm", "-y", is_flag=True, help="Skip confirmation prompt")
@click.option("--work-dir", "-w", type=click.Path(), default=None,
              help="Work directory (default: <project>/runs/<exp_id>)")
@click.option("--host", "-H", type=str, default=None,
              help="Target host name from runner_config.yaml")
@click.pass_context
def run(ctx: click.Context, exp_id: str, confirm: bool, work_dir: Optional[str], host: Optional[str]) -> None:
    """Launch an experiment via the configured runner."""
    from .runners.factory import create_runner

    project = _find_project(ctx)
    led = Ledger(project / "ledger")
    project_config = _load_project_config(project)

    exp = led.get(exp_id)
    if not exp:
        console.print(f"[red]Experiment {exp_id} not found.[/red]")
        raise SystemExit(1)

    if exp.status not in (ExperimentStatus.DRAFT, ExperimentStatus.QUEUED):
        console.print(f"[red]Experiment {exp_id} is {exp.status.value}, cannot run.[/red]")
        raise SystemExit(1)

    # Show experiment summary
    console.print(Panel(
        f"[bold]{exp.id}[/bold]: {exp.hypothesis.statement}\n\n"
        f"Script: {exp.config.script}\n"
        f"Args: {json.dumps(exp.config.script_args, indent=2)}\n"
        + (f"Env: {exp.config.env}\n" if exp.config.env else "")
        + (f"\nEstimated: {exp.config.resources.estimated_hours}h "
           f"(~${exp.config.resources.estimated_cost_usd})"
           if exp.config.resources and exp.config.resources.estimated_hours else ""),
        title="Experiment to Launch",
    ))

    if exp.config_diff:
        console.print("\n[bold]Config changes from baseline:[/bold]")
        for key, change in exp.config_diff.items():
            console.print(f"  {key}: {change.get('from')} → {change.get('to')}")

    if not confirm:
        if not click.confirm("\nLaunch this experiment?"):
            console.print("[dim]Cancelled.[/dim]")
            return

    # Create runner
    runner = create_runner(project, host=host)

    # Work directory
    if work_dir:
        wdir = Path(work_dir)
    else:
        wdir = project / "runs" / exp_id
    wdir.mkdir(parents=True, exist_ok=True)

    # Transition to queued then running
    if exp.status == ExperimentStatus.DRAFT:
        exp = exp.transition(ExperimentStatus.QUEUED)
    exp = exp.transition(ExperimentStatus.RUNNING)

    # Launch
    try:
        run_id = runner.launch(exp, wdir)
        exp = exp.model_copy(update={"run_id": run_id})
        led.update(exp)

        console.print(f"\n[green]Launched![/green]")
        console.print(f"  Run ID: {run_id}")
        console.print(f"  Work dir: {wdir}")
        console.print(f"\n  Check status: daedalus poll {exp_id}")
        console.print(f"  View logs:    daedalus logs {exp_id}")
        console.print(f"  Cancel:       daedalus cancel {exp_id}")
    except Exception as e:
        exp = exp.model_copy(update={"status": ExperimentStatus.FAILED})
        led.update(exp)
        console.print(f"[red]Launch failed: {e}[/red]")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# run-batch — launch multiple experiments across hosts
# ---------------------------------------------------------------------------

@cli.command("run-batch")
@click.argument("exp_ids", nargs=-1, required=False)
@click.option("--all-draft", is_flag=True, help="Launch all DRAFT experiments")
@click.option("--host", "-H", type=str, multiple=True,
              help="Target hosts (round-robin). Repeatable: -H gpu1 -H gpu2")
@click.option("--confirm", "-y", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def run_batch(
    ctx: click.Context,
    exp_ids: tuple[str, ...],
    all_draft: bool,
    host: tuple[str, ...],
    confirm: bool,
) -> None:
    """Launch multiple experiments, optionally distributed across hosts.

    Examples:

        # Launch specific experiments on specific hosts (round-robin)
        daedalus run-batch exp_001 exp_002 exp_003 -H gpu-a100 -H gpu-h100

        # Launch all draft experiments on default runner
        daedalus run-batch --all-draft

        # Launch all draft experiments distributed across 3 hosts
        daedalus run-batch --all-draft -H gpu1 -H gpu2 -H gpu3
    """
    from .runners.factory import create_runner

    project = _find_project(ctx)
    led = Ledger(project / "ledger")

    # Gather experiments
    if all_draft:
        experiments = [e for e in led.all() if e.status == ExperimentStatus.DRAFT]
        if not experiments:
            console.print("[yellow]No DRAFT experiments to launch.[/yellow]")
            return
    elif exp_ids:
        experiments = []
        for eid in exp_ids:
            exp = led.get(eid)
            if not exp:
                console.print(f"[red]Experiment {eid} not found, skipping.[/red]")
                continue
            if exp.status not in (ExperimentStatus.DRAFT, ExperimentStatus.QUEUED):
                console.print(f"[yellow]{eid} is {exp.status.value}, skipping.[/yellow]")
                continue
            experiments.append(exp)
        if not experiments:
            console.print("[red]No valid experiments to launch.[/red]")
            raise SystemExit(1)
    else:
        console.print("[red]Provide experiment IDs or use --all-draft.[/red]")
        raise SystemExit(1)

    # Build host list for round-robin
    hosts = list(host) if host else [None]  # None = default runner

    # Show summary
    console.print(f"\n[bold]Batch launch: {len(experiments)} experiments[/bold]")
    for i, exp in enumerate(experiments):
        target = hosts[i % len(hosts)] or "default"
        console.print(f"  {exp.id} → {target}: {exp.hypothesis.statement[:60]}")

    if not confirm:
        if not click.confirm(f"\nLaunch {len(experiments)} experiments?"):
            console.print("[dim]Cancelled.[/dim]")
            return

    # Launch
    launched = []
    failed = []
    for i, exp in enumerate(experiments):
        target_host = hosts[i % len(hosts)]
        try:
            runner = create_runner(project, host=target_host)
            wdir = project / "runs" / exp.id
            wdir.mkdir(parents=True, exist_ok=True)

            if exp.status == ExperimentStatus.DRAFT:
                exp = exp.transition(ExperimentStatus.QUEUED)
            exp = exp.transition(ExperimentStatus.RUNNING)

            run_id = runner.launch(exp, wdir)
            exp = exp.model_copy(update={"run_id": run_id})
            led.update(exp)

            launched.append((exp.id, target_host or "default", run_id))
            console.print(f"  [green]✓[/green] {exp.id} → {target_host or 'default'}")
        except Exception as e:
            exp = exp.model_copy(update={"status": ExperimentStatus.FAILED})
            led.update(exp)
            failed.append((exp.id, str(e)))
            console.print(f"  [red]✗[/red] {exp.id}: {e}")

    # Summary
    console.print(f"\n[bold]Results: {len(launched)} launched, {len(failed)} failed[/bold]")
    if launched:
        console.print(f"\n  Check all: daedalus watch --all")


# ---------------------------------------------------------------------------
# poll
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("exp_id")
@click.option("--host", "-H", type=str, default=None,
              help="Target host name from runner_config.yaml")
@click.pass_context
def poll(ctx: click.Context, exp_id: str, host: Optional[str]) -> None:
    """Check the status of a running experiment."""
    from .runners.factory import create_runner

    project = _find_project(ctx)
    led = Ledger(project / "ledger")

    exp = led.get(exp_id)
    if not exp:
        console.print(f"[red]Experiment {exp_id} not found.[/red]")
        raise SystemExit(1)

    if not exp.run_id:
        console.print(f"[yellow]Experiment {exp_id} has no run_id (not launched yet).[/yellow]")
        return

    runner = create_runner(project, host=host)
    status = runner.poll(exp.run_id)

    status_styles = {
        "queued": "yellow", "running": "blue",
        "completed": "green", "failed": "red",
    }
    style = status_styles.get(status.state, "")
    console.print(f"[{style}]{exp_id}: {status.state}[/]")

    if status.progress:
        console.print(f"  Progress: {status.progress}")
    if status.eta:
        console.print(f"  ETA: {status.eta}")
    if status.error:
        console.print(f"  [red]Error: {status.error}[/red]")

    # Update experiment status if terminal
    if status.state == "completed" and exp.status == ExperimentStatus.RUNNING:
        exp = exp.transition(ExperimentStatus.COMPLETED)
        # Try to fetch results
        try:
            results = runner.fetch_results(exp.run_id)
            exp = exp.model_copy(update={"results": results})
            console.print("\n[green]Results fetched:[/green]")
            for eval_name, metrics in results.items():
                for k, v in metrics.items():
                    console.print(f"  {eval_name}.{k} = {v}")
        except Exception as e:
            console.print(f"[yellow]Could not fetch results: {e}[/yellow]")
        led.update(exp)

    elif status.state == "failed" and exp.status == ExperimentStatus.RUNNING:
        exp = exp.transition(ExperimentStatus.FAILED)
        exp = exp.model_copy(update={"notes": exp.notes + f"\nFailed: {status.error or 'unknown'}"})
        led.update(exp)


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("exp_id")
@click.option("--tail", "-n", type=int, default=50, help="Number of lines")
@click.option("--host", "-H", type=str, default=None,
              help="Target host name from runner_config.yaml")
@click.pass_context
def logs(ctx: click.Context, exp_id: str, tail: int, host: Optional[str]) -> None:
    """View logs of a running or completed experiment."""
    from .runners.factory import create_runner

    project = _find_project(ctx)
    led = Ledger(project / "ledger")

    exp = led.get(exp_id)
    if not exp:
        console.print(f"[red]Experiment {exp_id} not found.[/red]")
        raise SystemExit(1)

    if not exp.run_id:
        console.print(f"[yellow]No run_id for {exp_id}.[/yellow]")
        return

    runner = create_runner(project, host=host)
    log_output = runner.logs(exp.run_id, tail=tail)
    console.print(log_output)


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("exp_id")
@click.option("--confirm", "-y", is_flag=True, help="Skip confirmation")
@click.option("--host", "-H", type=str, default=None,
              help="Target host name from runner_config.yaml")
@click.pass_context
def cancel(ctx: click.Context, exp_id: str, confirm: bool, host: Optional[str]) -> None:
    """Cancel a running experiment."""
    from .runners.factory import create_runner

    project = _find_project(ctx)
    led = Ledger(project / "ledger")

    exp = led.get(exp_id)
    if not exp:
        console.print(f"[red]Experiment {exp_id} not found.[/red]")
        raise SystemExit(1)

    if exp.status != ExperimentStatus.RUNNING:
        console.print(f"[yellow]Experiment {exp_id} is {exp.status.value}, not running.[/yellow]")
        return

    if not exp.run_id:
        console.print(f"[yellow]No run_id for {exp_id}.[/yellow]")
        return

    if not confirm:
        if not click.confirm(f"Cancel experiment {exp_id}?"):
            return

    runner = create_runner(project, host=host)
    runner.cancel(exp.run_id)

    exp = exp.transition(ExperimentStatus.ABANDONED)
    exp = exp.model_copy(update={"notes": exp.notes + "\nCancelled by user."})
    led.update(exp)

    console.print(f"[yellow]Experiment {exp_id} cancelled.[/yellow]")


# ---------------------------------------------------------------------------
# agent (autonomous research loop)
# ---------------------------------------------------------------------------

@cli.group()
def agent() -> None:
    """AI research agent commands."""
    pass


@agent.command("reason")
@click.pass_context
def agent_reason(ctx: click.Context) -> None:
    """Ask the agent to analyze state and propose the next hypothesis."""
    from .agent.loop import ResearchAgent, AgentMode

    project = _find_project(ctx)
    agent = ResearchAgent(project, mode=AgentMode.SINGLE_STEP)

    console.print("[bold]Agent reasoning...[/bold]\n")
    result = agent.reason()
    console.print(Markdown(result))


@agent.command("design")
@click.option("--hypothesis", "-h", type=str, default=None,
              help="Specific hypothesis to design for")
@click.pass_context
def agent_design(ctx: click.Context, hypothesis: Optional[str]) -> None:
    """Ask the agent to design the next experiment."""
    from .agent.loop import ResearchAgent, AgentMode

    project = _find_project(ctx)
    agent = ResearchAgent(project, mode=AgentMode.SINGLE_STEP)

    console.print("[bold]Agent designing experiment...[/bold]\n")
    result = agent.design(hypothesis=hypothesis)
    console.print(Markdown(result))


@agent.command("reflect")
@click.argument("exp_id")
@click.pass_context
def agent_reflect(ctx: click.Context, exp_id: str) -> None:
    """Ask the agent to analyze results of an experiment."""
    from .agent.loop import ResearchAgent, AgentMode

    project = _find_project(ctx)
    agent = ResearchAgent(project, mode=AgentMode.SINGLE_STEP)

    console.print(f"[bold]Agent reflecting on {exp_id}...[/bold]\n")
    result = agent.reflect(exp_id)
    console.print(Markdown(result))


@agent.command("plan")
@click.pass_context
def agent_plan(ctx: click.Context) -> None:
    """Ask the agent to create a structured research plan."""
    from .agent.loop import ResearchAgent, AgentMode

    project = _find_project(ctx)
    agent_obj = ResearchAgent(project, mode=AgentMode.SINGLE_STEP)

    console.print("[bold]Agent creating research plan...[/bold]\n")
    result = agent_obj.plan()
    console.print(Markdown(result))


@agent.command("loop")
@click.option("--max-experiments", "-n", type=int, default=3,
              help="Max experiments to run")
@click.option("--poll-interval", type=int, default=300,
              help="Seconds between polls (default: 300)")
@click.option("--autonomous", is_flag=True,
              help="Run without human confirmation")
@click.option("--plan/--no-plan", "use_plan", default=False,
              help="Follow research plan (creates one if missing)")
@click.option("--model", type=str, default=None,
              help="Claude model to use")
@click.pass_context
def agent_loop(
    ctx: click.Context,
    max_experiments: int,
    poll_interval: int,
    autonomous: bool,
    use_plan: bool,
    model: Optional[str],
) -> None:
    """Run the autonomous research loop."""
    from .agent.loop import ResearchAgent, AgentMode

    project = _find_project(ctx)
    mode = AgentMode.AUTONOMOUS if autonomous else AgentMode.INTERACTIVE

    kwargs: dict[str, Any] = {"mode": mode}
    if model:
        kwargs["model"] = model

    agent_obj = ResearchAgent(project, **kwargs)

    def on_step(step: str, detail: str) -> None:
        styles = {
            "REASON": "blue", "DESIGN_DONE": "cyan", "LAUNCHING": "yellow",
            "LAUNCHED": "green", "POLL": "dim", "REFLECTING": "magenta",
            "LOOP_DONE": "bold green", "FAILED": "red",
            "PLAN": "bold blue", "PLAN_DONE": "bold cyan",
            "PLAN_STEP": "bold yellow", "PLAN_COMPLETE": "bold green",
        }
        style = styles.get(step, "")
        console.print(f"[{style}][{step}][/] {detail[:120]}")

    plan_label = " with plan" if use_plan else ""
    console.print(f"[bold]Starting research loop ({mode.value}{plan_label}, max {max_experiments} experiments)[/bold]\n")

    exp_ids = agent_obj.run_loop(
        max_experiments=max_experiments,
        poll_interval=poll_interval,
        on_step=on_step,
        use_plan=use_plan,
    )

    console.print(f"\n[bold green]Done! Experiments: {exp_ids}[/bold green]")


cli.add_command(agent)


# ---------------------------------------------------------------------------
# plan — research plan management
# ---------------------------------------------------------------------------

@cli.group("plan")
def plan_group() -> None:
    """Research plan commands."""
    pass


@plan_group.command("show")
@click.pass_context
def plan_show(ctx: click.Context) -> None:
    """Show the current research plan."""
    from .agent.plan import PlanManager

    project = _find_project(ctx)
    manager = PlanManager(project / "ledger")
    plan = manager.load()

    if plan is None:
        console.print("[dim]No plan exists. Use 'daedalus agent plan' to create one.[/dim]")
        return

    console.print(f"\n[bold]Research Plan[/bold]: {plan.goal}")
    console.print(f"Version {plan.version} — {plan.updated_at.strftime('%Y-%m-%d %H:%M UTC')}\n")

    from rich.table import Table

    table = Table(show_header=True)
    table.add_column("ID", style="bold")
    table.add_column("P", justify="center")
    table.add_column("Status")
    table.add_column("Description")
    table.add_column("Experiment")
    table.add_column("Notes", max_width=40)

    status_styles = {
        "pending": "white", "running": "yellow", "done": "green",
        "skipped": "dim", "blocked": "red",
    }

    for step in plan.steps:
        style = status_styles.get(step.status.value, "")
        table.add_row(
            step.id,
            str(step.priority),
            f"[{style}]{step.status.value}[/{style}]",
            step.description,
            step.experiment_id or "",
            step.notes[:40] if step.notes else "",
        )

    console.print(table)

    # Summary
    done = sum(1 for s in plan.steps if s.status.value == "done")
    total = len(plan.steps)
    console.print(f"\nProgress: {done}/{total} done")

    next_step = manager.next_actionable()
    if next_step:
        console.print(f"[bold]Next[/bold]: {next_step.id} — {next_step.description}")


@plan_group.command("next")
@click.pass_context
def plan_next(ctx: click.Context) -> None:
    """Show the next actionable step."""
    from .agent.plan import PlanManager

    project = _find_project(ctx)
    manager = PlanManager(project / "ledger")
    step = manager.next_actionable()

    if step is None:
        console.print("[dim]No actionable steps (plan complete or doesn't exist).[/dim]")
        return

    console.print(f"\n[bold]{step.id}[/bold] (Priority {step.priority}): {step.description}")
    console.print(f"  Rationale: {step.rationale}")
    console.print(f"  Expected: {step.expected_outcome}")
    if step.depends_on:
        console.print(f"  Depends on: {', '.join(step.depends_on)}")


cli.add_command(plan_group)


# ---------------------------------------------------------------------------
# sync — upload project files to remote
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--host", "-H", default=None, help="Target host (from runner_config.yaml)")
@click.option("--files", "-f", multiple=True,
              help="Specific files/dirs to sync (default: scripts + requirements)")
@click.pass_context
def sync(ctx: click.Context, host: Optional[str], files: tuple) -> None:
    """Sync project files to remote host.

    Uploads training scripts, requirements, and config files to the remote
    work directory so experiments can run.

    \b
    Examples:
        daedalus sync                         # sync default files
        daedalus sync -H gpu-a100             # sync to specific host
        daedalus sync -f train.py -f data/    # sync specific files
    """
    from .runners.factory import create_runner
    from .runners.ssh import SSHRunner

    project = _find_project(ctx)
    runner = create_runner(project, host=host)

    if not isinstance(runner, SSHRunner):
        console.print("[red]Sync is only needed for SSH runners. Local runner uses files in place.[/red]")
        return

    config = _load_project_config(project)

    if files:
        # Sync specific files
        for f in files:
            src = project / f
            if not src.exists():
                console.print(f"[yellow]Skipping {f} (not found)[/yellow]")
                continue
            remote_dir = runner.config.remote_work_dir
            console.print(f"  Syncing {f} → {runner.config.host}:{remote_dir}/")
            runner.sync_files(src if src.is_dir() else src.parent, remote_dir)
        console.print("[green]Done.[/green]")
        return

    # Default: sync everything relevant
    remote_dir = runner.config.remote_work_dir
    console.print(f"Syncing project to {runner.config.host}:{remote_dir}/\n")

    # Collect files to sync
    sync_items: list[str] = []

    # Scripts from registry
    scripts = config.get("scripts", {})
    for name, spec in scripts.items():
        script_path = spec.get("path", "")
        if script_path and (project / script_path).exists():
            sync_items.append(script_path)

    # Requirements file
    stack = config.get("stack", {})
    req_file = stack.get("requirements", "requirements.txt")
    if (project / req_file).exists():
        sync_items.append(req_file)

    # daedalus.yaml itself
    sync_items.append("daedalus.yaml")

    if not sync_items:
        console.print("[yellow]No files to sync. Add scripts to daedalus.yaml.[/yellow]")
        return

    # Create temp dir with just the files we need and sync that
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for item in sync_items:
            src = project / item
            dst = tmp_path / item
            if src.is_dir():
                shutil.copytree(src, dst)
            elif src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            console.print(f"  [dim]{item}[/dim]")

        runner.sync_files(tmp_path, remote_dir)

    console.print(f"\n[green]Synced {len(sync_items)} items to {runner.config.host}:{remote_dir}/[/green]")


# ---------------------------------------------------------------------------
# setup-env — create/update environment on remote
# ---------------------------------------------------------------------------

@cli.command("setup-env")
@click.option("--host", "-H", default=None, help="Target host (from runner_config.yaml)")
@click.pass_context
def setup_env(ctx: click.Context, host: Optional[str]) -> None:
    """Set up Python environment on remote host.

    Runs the env_setup commands from daedalus.yaml on the remote host.
    If no env_setup is defined, installs requirements.txt with pip.

    \b
    Examples:
        daedalus setup-env                    # use env_setup from daedalus.yaml
        daedalus setup-env -H gpu-a100        # on specific host
    """
    from .runners.factory import create_runner
    from .runners.ssh import SSHRunner

    project = _find_project(ctx)
    runner = create_runner(project, host=host)

    if not isinstance(runner, SSHRunner):
        console.print("[yellow]setup-env is for remote hosts only.[/yellow]")
        return

    config = _load_project_config(project)
    stack = config.get("stack", {})
    remote_dir = runner.config.remote_work_dir

    env_setup = stack.get("env_setup", "").strip()

    if env_setup:
        # Use custom setup commands
        console.print(f"Running env_setup on {runner.config.host}...\n")
        for line in env_setup.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            console.print(f"  [dim]$ {line}[/dim]")

        result = runner._run_ssh(
            f"cd {remote_dir} && {env_setup}",
            timeout=600,
        )
    else:
        # Default: pip install requirements
        req_file = stack.get("requirements", "requirements.txt")
        python = runner.config.python_path

        console.print(f"Installing {req_file} on {runner.config.host}...\n")
        result = runner._run_ssh(
            f"cd {remote_dir} && {python} -m pip install -r {req_file}",
            timeout=600,
        )

    if result.returncode == 0:
        console.print(f"\n[green]Environment ready on {runner.config.host}.[/green]")
        if result.stdout.strip():
            # Show last few lines of output
            lines = result.stdout.strip().splitlines()
            for line in lines[-5:]:
                console.print(f"  [dim]{line}[/dim]")
    else:
        console.print(f"\n[red]Setup failed on {runner.config.host}:[/red]")
        console.print(result.stderr or result.stdout)


# ---------------------------------------------------------------------------
# watch (background daemon)
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("exp_id", required=False, default=None)
@click.option("--all", "wait_all", is_flag=True,
              help="Wait for ALL running experiments (default: first to finish)")
@click.option("--poll-interval", type=int, default=300,
              help="Seconds between status polls (default: 300)")
@click.option("--log-interval", type=int, default=10,
              help="Seconds between log tails (default: 10)")
@click.option("--stream", is_flag=True,
              help="Output JSONL events to stdout (for Claude Code)")
@click.option("--verbose", "-v", is_flag=True,
              help="Print human-readable events to stderr")
@click.option("--json-output", is_flag=True,
              help="Output final results as JSON")
@click.option("--stall-timeout", type=int, default=1800,
              help="Alert if no log output for N seconds (default: 1800)")
@click.pass_context
def watch(
    ctx: click.Context,
    exp_id: Optional[str],
    wait_all: bool,
    poll_interval: int,
    log_interval: int,
    stream: bool,
    verbose: bool,
    json_output: bool,
    stall_timeout: int,
) -> None:
    """Watch experiments — monitor status, tail logs, detect anomalies.

    A lightweight monitor — no AI decisions. The brain is Claude Code,
    which uses Daedalus MCP tools + code editing to analyze and iterate.

    In --stream mode, emits JSONL events to stdout:
    - METRIC:    parsed training metrics (loss, accuracy, lr)
    - PROGRESS:  epoch/step updates
    - ALERT:     anomalies (NaN, OOM, loss spike, stall)
    - COMPLETED: experiment finished
    - FAILED:    experiment failed

    \b
    Examples:
        daedalus watch exp_007                    # wait with pretty output
        daedalus watch exp_007 --stream           # JSONL events for Claude Code
        daedalus watch --all --verbose            # all experiments, human-readable
        daedalus watch --poll-interval 60         # check status every minute
        daedalus watch exp_007 --log-interval 5   # tail logs every 5 seconds
    """
    from .watcher import ExperimentWatcher

    project = _find_project(ctx)

    # If neither stream nor verbose, default to pretty console output
    use_console = not stream and not json_output

    def on_event(event: dict) -> None:
        if not use_console:
            return
        etype = event.get("event", "?")
        exp = event.get("exp_id", "")
        detail = event.get("detail", "")
        styles = {
            "START": "bold blue", "WATCHING": "blue",
            "PROGRESS": "cyan", "METRIC": "cyan",
            "COMPLETED": "bold green", "FAILED": "bold red",
            "ALERT": "bold yellow", "NO_RUNNING": "yellow",
            "POLL_ERROR": "red",
        }
        style = styles.get(etype, "")
        import time as _time
        ts = _time.strftime("%H:%M:%S")
        console.print(f"[dim]{ts}[/dim] [{style}][{etype}][/] {exp} {detail[:150]}")

    watcher = ExperimentWatcher(
        project,
        poll_interval=poll_interval,
        log_poll_interval=log_interval,
        stream=stream,
        verbose=verbose,
        on_event=on_event,
        stall_timeout=stall_timeout,
    )

    if use_console:
        console.print(f"[bold]Watching experiments (status every {poll_interval}s, logs every {log_interval}s)[/bold]")
        console.print("[dim]Press Ctrl+C to stop.[/dim]\n")

    if exp_id:
        result = watcher.wait_for(exp_id)
    else:
        result = watcher.watch_all(wait_all=wait_all)

    if json_output or stream:
        # In stream mode, final result was already emitted as events
        # But also output the final summary
        if json_output:
            console.print_json(json.dumps(result, default=str))
    elif isinstance(result, dict):
        _print_watch_result(result)
    elif isinstance(result, list):
        for r in result:
            _print_watch_result(r)


def _print_watch_result(result: dict) -> None:
    """Pretty-print a watch result."""
    exp_id = result.get("exp_id", "?")
    status = result.get("status", "?")

    if status == "completed":
        console.print(f"\n[bold green]{exp_id} completed![/bold green]")
        results = result.get("results")
        if results:
            for eval_name, metrics in results.items():
                for k, v in metrics.items():
                    console.print(f"  {eval_name}.{k} = {v}")
        console.print("\n[dim]Use Claude Code or 'daedalus agent reflect' to analyze.[/dim]")
    elif status == "failed":
        console.print(f"\n[bold red]{exp_id} failed![/bold red]")
        if result.get("error"):
            console.print(f"  Error: {result['error']}")
    else:
        console.print(f"\n[yellow]{exp_id}: {status}[/yellow]")


# ---------------------------------------------------------------------------
# install-mcp — register Daedalus as Claude Code MCP server
# ---------------------------------------------------------------------------

@cli.command("install-mcp")
@click.option("--global", "global_", is_flag=True,
              help="Install globally in ~/.claude/settings.json (default: project-level)")
@click.option("--python", "python_path", default=None,
              help="Python executable path (default: current interpreter)")
@click.pass_context
def install_mcp(ctx: click.Context, global_: bool, python_path: str | None) -> None:
    """Register Daedalus as a Claude Code MCP server."""
    import sys as _sys

    python_bin = python_path or _sys.executable

    if global_:
        # Global: use "." so the server discovers the project from Claude Code's
        # workspace cwd (which changes per project). No fixed path.
        project_arg = "."
        settings_dir = Path.home() / ".claude"
        settings_file = settings_dir / "settings.json"
        scope = "global"
    else:
        # Project-level: use absolute path to this specific project
        project = _find_project(ctx)
        project_arg = str(project.resolve())
        settings_dir = project / ".claude"
        settings_file = settings_dir / "settings.json"
        scope = "project"

    # Read existing settings
    settings: dict = {}
    if settings_file.exists():
        try:
            settings = json.loads(settings_file.read_text())
        except json.JSONDecodeError:
            settings = {}

    # Add MCP server config
    if "mcpServers" not in settings:
        settings["mcpServers"] = {}

    settings["mcpServers"]["daedalus"] = {
        "command": python_bin,
        "args": ["-m", "daedalus.mcp_server", "--project", project_arg],
    }

    # Write settings
    settings_dir.mkdir(parents=True, exist_ok=True)
    settings_file.write_text(json.dumps(settings, indent=2) + "\n")

    # Also install plugin files (agents, commands, skills)
    _install_plugin_files(settings_dir)

    # Auto-approve daedalus MCP tools
    if "permissions" not in settings:
        settings["permissions"] = {}
    allow = settings["permissions"].setdefault("allow", [])
    if "mcp__daedalus__*" not in allow:
        allow.append("mcp__daedalus__*")
        settings_file.write_text(json.dumps(settings, indent=2) + "\n")

    console.print(f"\n[bold green]Daedalus installed ({scope})![/bold green]")
    console.print(f"  Settings:  {settings_file}")
    console.print(f"  Python:    {python_bin}")
    if global_:
        console.print("  Project:   auto-detect from workspace (daedalus.yaml)")
    else:
        console.print(f"  Project:   {project_arg}")
    console.print(f"  Agents:    {settings_dir / 'agents'}")
    console.print(f"  Commands:  {settings_dir / 'commands'}")
    console.print(f"  Skills:    {settings_dir / 'skills'}")
    console.print("\n[dim]Restart Claude Code to activate.[/dim]")


def _install_plugin_files(target_dir: Path) -> None:
    """Copy agents, commands, and skills to the target .claude directory."""
    import shutil

    # Find the package's plugin files (relative to this file)
    package_root = Path(__file__).resolve().parent.parent

    for subdir in ("agents", "commands", "skills"):
        src = package_root / subdir
        if not src.exists():
            continue

        dst = target_dir / subdir
        dst.mkdir(parents=True, exist_ok=True)

        if subdir == "skills":
            # Skills are directories containing SKILL.md
            for skill_dir in src.iterdir():
                if skill_dir.is_dir():
                    dst_skill = dst / skill_dir.name
                    dst_skill.mkdir(parents=True, exist_ok=True)
                    for f in skill_dir.iterdir():
                        shutil.copy2(f, dst_skill / f.name)
        else:
            # Agents and commands are .md files
            for f in src.glob("*.md"):
                shutil.copy2(f, dst / f.name)

    console.print("  [green]✓[/green] Plugin files installed (agents, commands, skills)")


# ---------------------------------------------------------------------------
# uninstall-mcp — remove Daedalus MCP server registration
# ---------------------------------------------------------------------------

@cli.command("uninstall-mcp")
@click.option("--global", "global_", is_flag=True,
              help="Remove from global ~/.claude/settings.json")
@click.pass_context
def uninstall_mcp(ctx: click.Context, global_: bool) -> None:
    """Remove Daedalus MCP server from Claude Code settings."""
    project = _find_project(ctx)

    if global_:
        settings_file = Path.home() / ".claude" / "settings.json"
    else:
        settings_file = project / ".claude" / "settings.json"

    if not settings_file.exists():
        console.print("[yellow]No settings file found.[/yellow]")
        return

    try:
        settings = json.loads(settings_file.read_text())
    except json.JSONDecodeError:
        console.print("[red]Invalid settings.json[/red]")
        return

    servers = settings.get("mcpServers", {})
    if "daedalus" not in servers:
        console.print("[yellow]Daedalus MCP server not registered.[/yellow]")
        return

    del servers["daedalus"]
    if not servers:
        del settings["mcpServers"]

    settings_file.write_text(json.dumps(settings, indent=2) + "\n")
    console.print("[bold green]Daedalus MCP server removed.[/bold green]")
    console.print("\n[dim]Restart Claude Code to apply.[/dim]")


if __name__ == "__main__":
    cli()
