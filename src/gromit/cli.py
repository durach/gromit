"""Command-line interface for Gromit."""

import json
from enum import Enum
from pathlib import Path
from typing import Annotated, TypeVar

import typer
from dotenv import load_dotenv
from rich.console import Console

from gromit.config import Device, ModelSize, TranscriptionConfig

load_dotenv()

app = typer.Typer(
    name="gromit",
    help="Privacy-first local AI transcription with speaker diarization.",
)
console = Console()


@app.callback()
def main() -> None:
    """Privacy-first local AI transcription with speaker diarization.

    A callback (even an empty one) keeps Typer in multi-command "group" mode so
    `transcribe` stays an explicit subcommand. Without it, a single-command app
    collapses and `gromit transcribe FILE` would parse `transcribe` as a path.
    """


def validate_input_files(paths: list[Path]) -> list[Path]:
    """Validate that all input files exist."""
    for path in paths:
        if not path.exists():
            raise typer.BadParameter(f"File not found: {path}")
        if not path.is_file():
            raise typer.BadParameter(f"Not a file: {path}")
    return paths


_EnumT = TypeVar("_EnumT", bound=Enum)


def parse_choice(value: str, choices: type[_EnumT], flag: str) -> _EnumT:
    """Map a CLI string onto an enum, or raise `typer.BadParameter`.

    `--model` and `--device` are declared as plain `str` (the enum's repr would
    clutter `--help`), so Typer does no validation for us and a typo would reach
    `ModelSize(value)` as a bare `ValueError` — a raw traceback, which is not
    what an expected failure is supposed to look like. Converting here keeps the
    promise that a bad flag prints `Error: …` and exits 1.
    """
    try:
        return choices(value)
    except ValueError:
        allowed = ", ".join(member.value for member in choices)
        raise typer.BadParameter(
            f"Invalid {flag}: {value!r}. Choose one of: {allowed}"
        ) from None


def resolve_file_list(list_path: Path) -> list[Path]:
    """Read file paths from a text file.

    Args:
        list_path: Path to text file containing one file path per line.
            Blank lines and lines starting with # are ignored.
            Relative paths are resolved against the list file's parent directory.

    Returns:
        Ordered list of resolved Path objects.

    Raises:
        typer.BadParameter: If file contains no valid entries.
    """
    base_dir = list_path.parent
    paths = []
    for line in list_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line)
        if p.is_absolute() or (len(line) >= 2 and line[1] == ":"):
            paths.append(p)
        else:
            paths.append(base_dir / p)
    if not paths:
        raise typer.BadParameter(f"File list contains no file entries: {list_path}")
    return paths


@app.command()
def transcribe(
    input_files: Annotated[
        list[Path] | None,
        typer.Argument(
            help="Audio or video file(s) to transcribe",
        ),
    ] = None,
    from_file: Annotated[
        Path | None,
        typer.Option("--from-file", "-f", help="Text file with list of input files (one per line)"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output file path (default: <input>.gromit.txt)"),
    ] = None,
    language: Annotated[
        str,
        typer.Option("--language", "-l", help="Language code: en, uk, ru, auto"),
    ] = "auto",
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model size: tiny, base, small, medium, large-v3"),
    ] = "large-v3",
    speakers: Annotated[
        int | None,
        typer.Option("--speakers", "-s", help="Expected number of speakers"),
    ] = None,
    device: Annotated[
        str,
        typer.Option(help="Device: auto, cuda, mps, cpu"),
    ] = "auto",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
    duration: Annotated[
        float | None,
        typer.Option("--duration", help="Process only first N seconds (for testing)"),
    ] = None,
    glossary: Annotated[
        list[Path] | None,
        typer.Option("--glossary", help="Glossary YAML for hotwords (repeatable)."),
    ] = None,
) -> None:
    """Transcribe audio/video file(s) with speaker diarization.

    Multiple files are concatenated in order and transcribed as one.
    """
    # Lazy import: pulling pyannote/torch at module top would print a
    # torchcodec UserWarning and objc duplicate-class noise even for
    # `gromit transcribe --help`, which doesn't need the audio pipeline.
    from gromit.orchestrator import Orchestrator

    # Resolve input files from either positional args or --from-file
    try:
        if from_file and input_files:
            console.print("[red]Error:[/red] Cannot use both positional files and --from-file")
            raise typer.Exit(code=1)

        if from_file:
            if not from_file.exists():
                console.print(f"[red]Error:[/red] File not found: {from_file}")
                raise typer.Exit(code=1)
            resolved_files = resolve_file_list(from_file)
        elif input_files:
            resolved_files = input_files
        else:
            console.print("[red]Error:[/red] Provide input files or use --from-file")
            raise typer.Exit(code=1)

        resolved_files = validate_input_files(resolved_files)
        model_size = parse_choice(model, ModelSize, "--model")
        device_choice = parse_choice(device, Device, "--device")
    except typer.BadParameter as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    # Build configuration
    config = TranscriptionConfig(
        input_paths=resolved_files,
        output_path=output,
        language=language,
        model_size=model_size,
        device=device_choice,
        num_speakers=speakers,
        verbose=verbose,
        max_duration=duration,
        from_file_path=from_file,
        glossary_paths=glossary or [],
    )

    # Run transcription
    try:
        orchestrator = Orchestrator(config)
        result = orchestrator.process()

        # Write output
        output_path = config.effective_output_path
        output_path.write_text(result)

        console.print(f"[green]Transcript saved to:[/green] {output_path}")

        json_path = config.json_output_path
        json_path.write_text(
            json.dumps(orchestrator.transcript_json(), ensure_ascii=False, indent=2)
        )
        console.print(f"[green]Structured JSON saved to:[/green] {json_path}")

    # Outermost CLI boundary. Every pipeline failure — ffmpeg, torch, pyannote,
    # faster-whisper, disk I/O — must reach the user as one red line and exit 1,
    # never as a traceback. Narrowing to GromitError would let third-party errors
    # escape unformatted, which is the opposite of what this handler is for.
    except Exception as e:  # noqa: BLE001 — user-facing catch-all; see above
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def crosscheck(
    gromit_json: Annotated[
        Path,
        typer.Argument(help="The <stem>.gromit.json transcript to check."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output flags.json path."),
    ],
    meet: Annotated[
        Path | None,
        typer.Option("--meet", help="Google Meet WebVTT to compare against (optional)."),
    ] = None,
    glossary: Annotated[
        list[Path] | None,
        typer.Option("--glossary", help="Glossary YAML for misheard matching (repeatable)."),
    ] = None,
) -> None:
    """Flag spans needing review: engine divergence, low confidence, mishearings."""
    from collections import Counter

    from gromit.crosscheck.core import run_crosscheck, write_flags_json
    from gromit.exceptions import GromitError

    try:
        spans = run_crosscheck(gromit_json, meet, glossary or [])
        write_flags_json(output, spans)
    except GromitError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    reason_counts = Counter(r for s in spans for r in s.reasons)
    console.print(
        f"[green]Wrote[/green] {output} — {len(spans)} spans ({dict(reason_counts)})"
    )


@app.command(name="glossary-merge")
def glossary_merge(
    corrections: Annotated[
        Path,
        typer.Argument(help="corrections.yaml exported from the review page."),
    ],
    glossary: Annotated[
        Path,
        typer.Option("--glossary", help="The per-project glossary.yaml to update in place."),
    ],
) -> None:
    """Fold review corrections into the glossary (idempotent, comment-preserving)."""
    from gromit.exceptions import GromitError
    from gromit.glossary_merge import load_corrections, merge_corrections

    try:
        summary = merge_corrections(glossary, load_corrections(corrections))
    except GromitError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    for canonical in summary.added_entries:
        console.print(f"[green]+ entry[/green] {canonical}")
    for canonical, heard in summary.added_misheard:
        if canonical not in summary.added_entries:
            console.print(f"[green]+ misheard[/green] {canonical} ← {heard}")
    console.print(
        f"[green]Merged[/green] {glossary} — "
        f"{len(summary.added_entries)} new, {len(summary.added_misheard)} misheard added, "
        f"{summary.unchanged} unchanged"
    )


@app.command()
def review(
    flags_json: Annotated[
        Path,
        typer.Argument(help="The flags.json produced by `gromit crosscheck`."),
    ],
    video: Annotated[
        Path,
        typer.Option("--video", help="The recording MP4 the flags refer to."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output review/ directory."),
    ],
    named: Annotated[
        Path | None,
        typer.Option("--named", help="nametag .named.vtt for speaker labels (optional)."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Only the top-N ranked spans (default: all)."),
    ] = None,
) -> None:
    """Build a self-contained review page: one video clip + correction box per span."""
    from gromit.exceptions import GromitError
    from gromit.review.core import run_review

    try:
        summary = run_review(flags_json, video, named, output, limit=limit)
    except GromitError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    console.print(
        f"[green]Wrote[/green] {summary['out_dir']}/index.html — "
        f"{summary['spans']} spans, {summary['clips_ok']} clips extracted"
    )


@app.command()
def nametag(
    folder: Annotated[
        Path | None,
        typer.Argument(help="Meeting folder holding the recording (.mp4) and a Google Meet "
                            "caption file (.vtt); first of each by name is used"),
    ] = None,
    video: Annotated[
        Path | None,
        typer.Option("--video", help="Recording path — use when it does not share a stem "
                                     "with the caption file. Requires --vtt."),
    ] = None,
    vtt: Annotated[
        Path | None,
        typer.Option("--vtt", help="Google Meet caption file. Requires --video."),
    ] = None,
    guest: Annotated[
        list[str] | None,
        typer.Option("--guest", "-g", help="Occasional attendee name (repeatable)."),
    ] = None,
    roster: Annotated[
        Path | None,
        typer.Option("--roster", help="roster.yaml of permanent members (optional)."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Print per-cue votes instead of a progress bar."),
    ] = False,
    early_stop: Annotated[
        bool,
        typer.Option("--early-stop/--no-early-stop", help="Stop a cue once a roster name holds a majority."),
    ] = True,
    keep_cache: Annotated[
        bool,
        typer.Option("--keep-cache", help="Keep the frame cache after the run "
                     "(cache is also auto-kept when cues need review)."),
    ] = False,
) -> None:
    """Tag Google Meet speakers from video: writes <stem>.named.vtt + .named.txt.

    Give a meeting FOLDER, and the alphabetically first .mp4 plus the first .vtt
    that is not a .named.vtt (this command's own output) are used. Where the two
    files do not share a stem — as with Meet's own exports, `… Recording.mp4`
    beside `… Recording-uk-asr.vtt` — name them with --video and --vtt instead.

    Each caption cue is labelled with the on-screen speaker. The .vtt is Google
    Meet's caption track, downloaded from the Drive player — gromit does not
    create it. Candidates come from --roster (its `permanent:` list) and/or
    --guest. macOS uses best-of-both EasyOCR + Apple Vision; elsewhere EasyOCR.
    """
    # Lazy import: defers cv2/torch/easyocr (and their startup chatter) so
    # `gromit nametag --help` stays instant.
    from gromit.nametag.roster import load_roster
    from gromit.nametag.run import attribute_meeting

    guests = list(guest or [])
    if (video is None) != (vtt is None):
        console.print("[red]Error:[/red] --video and --vtt must be given together")
        raise typer.Exit(code=1)
    if video is not None:
        # Explicit paths win; results land in FOLDER when given, else beside the video.
        out_dir = folder or video.parent
    elif folder is not None:
        mp4s = sorted(folder.glob("*.mp4"))
        vtts = sorted(p for p in folder.glob("*.vtt") if not p.name.endswith(".named.vtt"))
        if not mp4s or not vtts:
            console.print(f"[red]Error:[/red] need one .mp4 and one .vtt in {folder}")
            raise typer.Exit(code=1)
        video, vtt, out_dir = mp4s[0], vtts[0], folder
    else:
        console.print("[red]Error:[/red] give a meeting FOLDER or --video with --vtt")
        raise typer.Exit(code=1)
    candidates = [*(load_roster(roster).permanent if roster else []), *guests]
    if not candidates:
        console.print("[red]Error:[/red] no candidate names — pass --roster and/or --guest")
        raise typer.Exit(code=1)

    if verbose:
        def on_cue(i: int, total: int, name: str) -> None:
            console.print(f"[{i:5d}/{total}] {name}")

        summary = attribute_meeting(video, vtt, out_dir, candidates,
                                    early_stop=early_stop, keep_cache=keep_cache, on_cue=on_cue)
    else:
        from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(),
                      TextColumn("{task.completed}/{task.total}"), TimeRemainingColumn(),
                      console=console) as progress:
            task = progress.add_task(f"nametag {video.stem}", total=None)

            def on_cue(i: int, total: int, name: str) -> None:
                if progress.tasks[task].total is None:
                    progress.update(task, total=total)
                progress.update(task, advance=1)

            summary = attribute_meeting(video, vtt, out_dir, candidates,
                                        early_stop=early_stop, keep_cache=keep_cache, on_cue=on_cue)

    engine = "best-of-both (EasyOCR + Apple Vision)" if summary["use_vision"] else "EasyOCR only"
    console.print(f"[green]Wrote[/green] {summary['named_vtt']} (+ .named.txt) — "
                  f"{summary['cues']} cues, {engine}")
    if summary["kept"] and summary["needs_review"]:
        console.print(
            f"[yellow]{summary['needs_review']} cues need review[/yellow] — "
            f"either no on-screen name could be read (labelled Unknown in the "
            f"output) or the name read is not one of your candidates (kept "
            f"verbatim, never snapped to a roster entry). The sampled frames "
            f"are kept at {summary['cache_dir']} so you can check them by eye."
        )


if __name__ == "__main__":
    app()
