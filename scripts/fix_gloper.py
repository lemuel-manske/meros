from pathlib import Path
import sys


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if old not in source:
        if new in source:
            return source  # idempotent

        raise RuntimeError(f"Cannot locate upstream code for {label}; inspect upstream changes")

    if source.count(old) != 1:
        raise RuntimeError(f"Expected exactly one instance of {label}")

    return source.replace(old, new, 1)


def main(root: Path) -> None:
    model_path = root / "models.py"
    gloper_path = root / "gloper.py"

    model = model_path.read_text()

    model = replace_once(model, "class UNetColorCentroidLocal2d(nn.Module):", "class UNetColorCentroidLocal(nn.Module):", "model class")

    model = replace_once(model, "super(UNetColorCentroidLocal, self).__init__()", "super().__init__()", "model super")

    src = gloper_path.read_text()

    src = replace_once(src,
        "F.interpolate(fur_color, size=(1024, 1024), mode='nearest')",
        "F.interpolate(fur_color, size=input_image.shape[-2:], mode='nearest')",
        "dynamic foreground upsampling")

    src = replace_once(src,
        "F.interpolate(bg_color, size=(1024, 1024), mode='nearest')",
        "F.interpolate(bg_color, size=input_image.shape[-2:], mode='nearest')",
        "dynamic background upsampling")

    src = replace_once(src,
        '    optimizer_name = "AdamW"\n):',
        '    optimizer_name = "AdamW",\n    foreground_mask = None,\n):',
        "optional foreground mask argument")

    src = replace_once(src,
        '    bg_mask = get_bg_mask(image_o, pxl_size)\n',
        '    bg_mask = (get_bg_mask(image_o, pxl_size) if foreground_mask is None\n'
        '               else foreground_mask)\n',
        "foreground selection")

    src = replace_once(src,
        'binary_fur_mask = cv2.bitwise_not(binary_fur_mask)',
        'binary_fur_mask = 1.0 - binary_fur_mask',
        "torch-compatible polarity")

    src = replace_once(src,
        'base_dir = r""\nprocess_all_images(base_dir, method="gloper")',
        '# No automatic batch processing on import.',
        "import-time execution")

    model_path.write_text(model)

    gloper_path.write_text(src)

    print("Patched:", model_path, gloper_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/patch_gloper.py external/GloPER")

    main(Path(sys.argv[1]).resolve())
