# TODO: replace by env variables

_TRACKS_FOLDER = "/home/lkmliz/dev/meros/tracks"
_VIDEOS_FOLDER = "/home/lkmliz/dev/meros/videos"

VIDEOS = [
    "260113_S3_SC_Monoboia_0025_1#.mov",
    # "260113_S3_SC_Monoboia_0025_2#_3#.mov",
    # "260113_S3_SC_Monoboia_0026_4#.mov",
    # "260113_S3_SC_Monoboia_0039_Diversos#.mov",
    # "260113_S3_SC_Monoboia_0065_5#.mov",
    # "260113_S3_SC_Monoboia_0065_6#.mov",
    # "260113_S3_SC_Monoboia_0065_7#.mov",
    # "260113_S3_SC_Monoboia_0065_8#.mov",
    # "260113_S3_SC_Monoboia_0066_10#.mov",
    # "260113_S3_SC_Monoboia_0066_9#.mov",
    # "260113_S3_SC_Monoboia_0101_11#.mov",
    # "260113_S3_SC_Monoboia_0101_12#.mov",
    # "260113_S3_SC_Monoboia_0102_13#.mov",
    # "260113_S3_SC_Monoboia_0102_14#.mov",
    # "260113_S3_SC_Monoboia_0102_15#.mov",
    # "260113_S3_SC_Monoboia_0147_16#.mov",
    # "260113_S3_SC_Monoboia_0149_Diversos#.mov",
]


def path_to_track(vid_id):
    """
    Get the path to the track annotations for a given video ID.
    """
    wo_ext = vid_id.split(".")[0]
    return f"{_TRACKS_FOLDER}/{wo_ext}"


def path_to_vid(vid_id):
    """
    Get the path to a video given its ID.
    """
    return f"{_VIDEOS_FOLDER}/{vid_id}"


def get_vid_id(vid_path):
    """
    Extract the video ID from the video path.
    """
    return vid_path.split("/")[-1].split(".")[0]


def get_annotations(vid_id):
    """
    Get the path to the track annotations for a given video ID.
    """
    wo_ext = vid_id.split(".")[0]
    return f"{_TRACKS_FOLDER}/{wo_ext}/annotations.xml"
