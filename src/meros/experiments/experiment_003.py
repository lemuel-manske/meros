"""Exp-003: exploratory head-spot graphs with affine RANSAC."""
from __future__ import annotations

import argparse
import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.spatial import cKDTree


def load_frame(path: Path, index: int) -> np.ndarray:
    capture = cv.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise FileNotFoundError(path)
        capture.set(cv.CAP_PROP_POS_FRAMES, index)
        ok, image = capture.read()
        if not ok or image is None:
            raise ValueError(f"Cannot decode frame {index}: {path}")
        return image
    finally:
        capture.release()


def detect_spots(image: np.ndarray, roi: tuple[int, int, int, int] | None) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    x, y, w, h = roi if roi is not None else (0, 0, width, height)
    if not (0 <= x < width and 0 <= y < height and w > 0 and h > 0
            and x + w <= width and y + h <= height):
        raise ValueError(f"ROI outside image: {roi}")
    gray = cv.cvtColor(image[y:y+h, x:x+w], cv.COLOR_BGR2GRAY)
    gray = cv.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(gray)
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv.THRESH_BINARY_INV, 35, 8)
    count, _, stats, centers = cv.connectedComponentsWithStats(binary, 8)
    found = []
    for i in range(1, count):
        left, top, bw, bh, area = map(int, stats[i])
        if not (5 <= area <= max(10, int(w*h*.001)) and bw >= 2 and bh >= 2):
            continue
        if left == 0 or top == 0 or left+bw >= w or top+bh >= h:
            continue
        if not (.35 <= bw/bh <= 2.8 and .25 <= area/(bw*bh) <= .94):
            continue
        cx, cy = centers[i]
        found.append((float(cx+x), float(cy+y), area))
    found.sort(key=lambda item: (-item[2], item[1], item[0]))
    points = np.asarray([(a,b) for a,b,_ in found[:180]], np.float32).reshape(-1,2)
    annotated = image.copy()
    cv.rectangle(annotated, (x,y), (x+w,y+h), (255,180,0), 2)
    for px,py in points:
        cv.circle(annotated, (round(float(px)),round(float(py))),4,(0,255,255),1)
    return points, annotated


def descriptors(points: np.ndarray, k: int = 6) -> np.ndarray:
    """Relative neighbor distances: rotation/translation/scale, not affine invariant."""
    if len(points) < k+1:
        return np.empty((0,k), np.float32)
    distances, _ = cKDTree(points).query(points,k=k+1)
    neighbor = distances[:,1:].astype(np.float32)
    return neighbor/np.maximum(neighbor[:,-1:],1e-5)


def matches(a: np.ndarray, b: np.ndarray) -> list[tuple[int,int]]:
    da,db = descriptors(a),descriptors(b)
    if len(da) < 2 or len(db) < 2:
        return []
    distance,index = cKDTree(db).query(da,k=2)
    _,reverse = cKDTree(da).query(db,k=1)
    return [(i,int(index[i,0])) for i in range(len(da))
            if distance[i,0] < .88*max(distance[i,1],1e-10)
            and reverse[int(index[i,0])] == i]


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    pairs = matches(a,b)
    result = dict(source_spots=len(a),target_spots=len(b),tentative_matches=len(pairs),
                  ransac_attempted=False,inliers=0,inlier_ratio=None,
                  median_residual_px=None,source_coverage=0.,target_coverage=0.,
                  pairs=pairs,inlier_mask=[False]*len(pairs))
    if len(pairs) < 3:
        return result
    src = np.asarray([a[i] for i,j in pairs],np.float32)
    dst = np.asarray([b[j] for i,j in pairs],np.float32)
    matrix, mask = cv.estimateAffine2D(src,dst,method=cv.RANSAC,
                                        ransacReprojThreshold=5,maxIters=3000,
                                        confidence=.99,refineIters=10)
    result["ransac_attempted"] = True
    if matrix is None or mask is None:
        return result
    good = mask.ravel().astype(bool)
    residual = np.linalg.norm(src@matrix[:,:2].T + matrix[:,2] - dst,axis=1)
    n = int(good.sum())
    result.update(inliers=n,inlier_ratio=n/len(pairs),
                  median_residual_px=float(np.median(residual[good])) if n else None,
                  source_coverage=n/max(1,len(a)),target_coverage=n/max(1,len(b)),
                  inlier_mask=good.tolist())
    return result


def draw_matches(a: np.ndarray,b: np.ndarray,pa: np.ndarray,pb: np.ndarray,m: dict) -> np.ndarray:
    height = max(a.shape[0],b.shape[0])
    output = np.zeros((height,a.shape[1]+b.shape[1],3),dtype=np.uint8)
    output[:a.shape[0],:a.shape[1]] = a
    output[:b.shape[0],a.shape[1]:] = b
    for (i,j),valid in zip(m["pairs"],m["inlier_mask"],strict=True):
        p = tuple(np.rint(pa[i]).astype(int))
        q = tuple(np.rint(pb[j]).astype(int)+[a.shape[1],0])
        cv.line(output,p,q,(0,220,0) if valid else (40,80,220),1)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("data"))
    parser.add_argument("--output",type=Path)
    parser.add_argument("--roi-h01",nargs=4,type=int,metavar=("X","Y","W","H"))
    parser.add_argument("--roi-h02",nargs=4,type=int,metavar=("X","Y","W","H"))
    parser.add_argument("--delta",type=int,default=30)
    args = parser.parse_args()
    if args.delta <= 0:
        parser.error("--delta must be positive")
    videos = {v["video_id"]:v for v in json.loads(
        (args.data/"metadata/videos/videos.json").read_text())}
    people = json.loads((args.data/"metadata/individuals/individuals.json").read_text())
    selections = {t["video_id"]:(str(p["individual_id"]),t)
                  for p in people for t in p["tracks"] if t["video_id"] in ("h01","h02")}
    if set(selections) != {"h01","h02"}:
        raise ValueError("Need both h01 and h02 in individuals.json")
    if selections["h01"][0] == selections["h02"][0]:
        raise ValueError("h01 and h02 expected to be different individuals")
    output = args.output or Path("results/003")/(datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False)
    samples = {}
    for video_id in ("h01","h02"):
        track = selections[video_id][1]
        ref = int(track["reference_frame"])
        offset = ref+args.delta
        if offset > int(track["end_frame"]):
            offset = ref-args.delta
        if offset < int(track["start_frame"]):
            raise ValueError(f"No offset frame for {video_id}")
        roi = getattr(args,"roi_"+video_id)
        for label,index in (("reference",ref),("offset",offset)):
            frame = load_frame(args.data.parent/videos[video_id]["path"],index)
            # Mirrored metadata is documented but not automatically re-applied.
            points,annotated = detect_spots(frame,tuple(roi) if roi else None)
            if not cv.imwrite(str(output/f"{video_id}_{label}_spots.png"),annotated):
                raise IOError("Failed to write spot visualization")
            samples[video_id,label] = (frame,points)
    comparisons = [("h01","reference","h01","offset",True),
                   ("h02","reference","h02","offset",True),
                   ("h01","reference","h02","reference",False)]
    rows = []
    for va,fa,vb,fb,same in comparisons:
        a,pa = samples[va,fa]
        b,pb = samples[vb,fb]
        metrics = compare(pa,pb)
        image = draw_matches(a,b,pa,pb,metrics)
        if not cv.imwrite(str(output/f"{va}_{fa}_vs_{vb}_{fb}.png"),image):
            raise IOError("Failed to write matching visualization")
        rows.append(dict(source=f"{va}/{fa}",target=f"{vb}/{fb}",
                         same_individual=same,**{k:v for k,v in metrics.items()
                         if k not in ("pairs","inlier_mask")}))
    with (output/"metrics.csv").open("w",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = {"delta":args.delta,"roi_h01":args.roi_h01,"roi_h02":args.roi_h02,
                  "video_metadata":{k:{"mirrored":videos[k].get("mirrored",False)}
                                    for k in ("h01","h02")}}
    (output/"run.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print(f"Exp-003: wrote {len(rows)} comparisons to {output}")


if __name__ == "__main__":
    main()
