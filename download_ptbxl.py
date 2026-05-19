"""Unzip a pre-uploaded ptbxl.zip inside a Modal container.

Because modal volume put is slow with many small files, upload a single zip
and let this script unzip it in the container:

    modal volume put gdss-cache ~/Downloads/ptbxl.zip ptbxl.zip
    modal run download_ptbxl.py
"""

from __future__ import annotations

import modal_common

PTBXL_DEST = f"{modal_common.REMOTE_CACHE}/ptbxl"


@modal_common.app.function(
    image=modal_common.image,
    volumes={modal_common.REMOTE_CACHE: modal_common.cache_vol},
    timeout=600,
    cpu=2,
)
def unzip_ptbxl() -> None:
    """Unzip ptbxl.zip from the cache volume into PTBXL_DEST."""
    import shutil
    import zipfile
    from pathlib import Path

    zip_path = Path(modal_common.REMOTE_CACHE) / "ptbxl.zip"
    dest = Path(PTBXL_DEST)

    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} not found in volume.\n"
            "Upload it first:\n"
            "  modal volume put gdss-cache ~/Downloads/ptbxl.zip ptbxl.zip"
        )

    if (dest / "ptbxl_database.csv").exists():
        print("PTB-XL already unzipped — nothing to do.")
        return

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Unzipping {zip_path} → {dest} …")

    with zipfile.ZipFile(zip_path) as zf:
        top = Path(zf.namelist()[0]).parts[0]
        members = [m for m in zf.infolist() if not m.is_dir()]
        for i, member in enumerate(members):
            target = dest / Path(member.filename).relative_to(top)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            if i % 2000 == 0:
                print(f"  {i}/{len(members)} files…")

    zip_path.unlink()
    print(f"PTB-XL ready at {dest}")
    modal_common.cache_vol.commit()


@modal_common.app.local_entrypoint(name="download")
def main() -> None:
    unzip_ptbxl.remote()
