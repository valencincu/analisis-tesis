import functools
from pathlib import Path
from shutil import copyfile

def create_output_directory(video, import_dir, export_dir):
    job_id = input("Enter job identifier: ", )

    save_dir = export_dir / f"{import_dir.name} - {job_id}"
    save_dir.mkdir(parents=True, exist_ok=False)

    metadata_path = next(video.directory_path.glob("*.cih"))
    copyfile(metadata_path, save_dir / "metadata.cih")
    
    return save_dir, save_dir / "notes.txt"

def write_analysis_notes(path_dictionary, parameter_dictionary, notes_path):
    user_input = input("NOTES:\n")
    notes = f"NOTES:\n{user_input}"
    paths = "PATHS\n" + "\n".join(
        f"{name} : {path}" for name, path in path_dictionary.items()
    )
    parameters = "PARAMETERS\n" + "\n".join(
        f"{name} : {parameter}" for name, parameter in parameter_dictionary.items()
    )
    script = Path(path_dictionary["script_path"]).resolve().read_text()
    notes_path.write_text("\n---\n".join((notes, paths, parameters, script)))