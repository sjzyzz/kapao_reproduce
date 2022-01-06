from pathlib import Path


def attempt_download(file, repo='ultralytics/yolov5'):
    file = Path(str(file).strip().replace("'", ''))

    if not file.exists():
        pass
    return str(file)