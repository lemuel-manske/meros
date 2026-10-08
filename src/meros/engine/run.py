from .engine import pipeline


if __name__ == "__main__":
    pipeline.run("masked_crops")  # runs a single pipeline stage
    # pipeline.run_all()
