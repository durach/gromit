"""CLI interface for Gromit."""

import click
from pathlib import Path
from rich.console import Console
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn
)
import sys
import os
from dotenv import load_dotenv

from gromit import __version__

# Load environment variables
load_dotenv()

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="gromit")
def cli():
    """Gromit – Local AI Tool for Transcribing Meetings"""
    pass


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output file path. Defaults to input_transcript.txt",
)
@click.option(
    "-l",
    "--language",
    default=os.getenv("DEFAULT_LANGUAGE", "en"),
    help=f"Language code (e.g., 'en' for English, 'uk' for Ukrainian). Default: {os.getenv('DEFAULT_LANGUAGE', 'en')}",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda", "mps"]),
    default=os.getenv("DEFAULT_DEVICE", "auto"),
    help=f"Device to use for inference. Default: {os.getenv('DEFAULT_DEVICE', 'auto')}",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose logging",
)
@click.option(
    "--max-seconds",
    type=float,
    help="Only transcribe first X seconds (for debugging/testing)",
)
def transcribe(input_file, output, language, device, verbose, max_seconds):
    """Transcribe audio file with speaker diarization."""
    
    # Set output path if not provided
    if output is None:
        output = input_file.parent / f"{input_file.stem}_transcript.txt"
    
    console.print(f"[bold blue]Gromit v{__version__}[/bold blue]")
    console.print(f"Input: {input_file}")
    console.print(f"Output: {output}")
    console.print(f"Language: {language}")
    
    try:
        # Import here to avoid slow startup
        from gromit.transcriber import transcribe_audio
        from gromit.diarizer import diarize_audio
        from gromit.audio_utils import convert_to_wav, get_audio_duration
        import soundfile as sf
        import tempfile
        
        # Convert to WAV if needed (for MP4, MP3, etc.)
        converted_file, is_temporary = convert_to_wav(str(input_file))
        working_file = Path(converted_file)
        
        # Get audio duration for progress estimation
        full_duration = get_audio_duration(str(input_file))
        
        # Use max_seconds if specified
        if max_seconds:
            duration = min(max_seconds, full_duration)
            console.print(f"Duration: {full_duration:.1f} seconds ({full_duration/60:.1f} minutes)")
            console.print(f"[yellow]DEBUG MODE: Processing only first {duration:.1f} seconds[/yellow]")
        else:
            duration = full_duration
            console.print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
        # Estimate processing time
        estimated_time = duration * 0.15  # Rough estimate: 15% of audio duration
        console.print(f"[dim]Estimated processing time: ~{estimated_time/60:.0f} minutes[/dim]")
        console.print()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("elapsed"),
            console=console,
        ) as progress:
            # Step 1: Diarization
            task1 = progress.add_task(
                "[cyan]Analyzing speakers (this may take a few minutes)...", 
                total=None  # Indeterminate progress
            )
            
            speaker_segments = diarize_audio(working_file, device=device, max_seconds=max_seconds)
            progress.update(task1, description="[green]✓ Speaker diarization complete")
            
            # Step 2: Transcription
            task2 = progress.add_task(
                f"[green]Transcribing {duration/60:.1f} minutes of audio...", 
                total=None  # Use indeterminate progress since we can't track it
            )
            
            transcription = transcribe_audio(
                working_file, 
                language=language, 
                device=device,
                verbose=verbose,
                max_seconds=max_seconds
            )
            progress.update(task2, description="[green]✓ Transcription complete")
            
            # Step 3: Merge and format
            task3 = progress.add_task("[yellow]Formatting output...", total=None)
            from gromit.formatter import format_conversation
            formatted_text = format_conversation(transcription, speaker_segments)
            progress.update(task3, description="[green]✓ Formatting complete")
        
        # Write output
        output.write_text(formatted_text, encoding="utf-8")
        console.print(f"[green]✓[/green] Transcription saved to {output}")
        
        # Clean up temporary file if created
        if is_temporary:
            os.unlink(converted_file)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        if verbose:
            console.print_exception()
        sys.exit(1)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()