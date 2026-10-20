from src.meros.processing.extract_frames import run
from src.meros.project import Project

if __name__ == "__main__":
    run(Project.open())
