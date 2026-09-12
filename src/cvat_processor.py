from typing import TypedDict

import xml.etree.ElementTree as ET


class Annotation(TypedDict):
    frame_number: int
    image_name: str
    label: str
    xtl: float
    ytl: float
    xbr: float
    ybr: float


def parse(xml_file_path) -> list[Annotation]:
    """
    Parses a CVAT XML annotation file and returns a list of annotations.

    Args:
        xml_file (str): Path to the CVAT XML annotation file.
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    annotations = []

    for image in root.findall('image'):
        image_name: str = image.get('name') or ''

        for box in image.findall('box'):
            label = box.get('label') or ''

            xtl = float(box.get('xtl') or 0)
            ytl = float(box.get('ytl') or 0)
            xbr = float(box.get('xbr') or 0)
            ybr = float(box.get('ybr') or 0)

            annotations.append({
                'frame_number': int(image_name.split('_')[-1]),
                'image_name': image_name,
                'label': label,
                'xtl': xtl,
                'ytl': ytl,
                'xbr': xbr,
                'ybr': ybr
            })

    return annotations
