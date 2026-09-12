import cv2 as cv
import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor

from src.artifacts import (
    get_vids,
    path_to_frame,
    path_to_frames,
    path_to_vis,
)
from src.consts import (
    SAM2_CHECKPOINT,
    SAM2_MODEL_CONFIG,
)
from src.metadata import (
    VisMetadata,
    persist_vis_metadata,
)


if __name__ == "__main__":
    """
    This script initializes the SAM2 video predictor,
    allows the user to select a bounding box for the subject in the first frame,
    and then propagates the mask through the video frames.

    The resulting visualizations are saved to disk.
    """

    predictor = build_sam2_video_predictor(
        SAM2_MODEL_CONFIG,
        SAM2_CHECKPOINT,
        device="cuda",
        dtype=torch.bfloat16,
    )

    for vid_id in get_vids():
        frame_idx = 0
        frame_path = path_to_frame(vid_id, frame_idx)

        print(f"Initializing inference state for video: {vid_id}, frame: {frame_idx}, path: {frame_path}")


        # use gpu for inference
        inference_state = predictor.init_state(
            video_path=path_to_frames(vid_id),
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
        )


        print("Inference state initialized successfully.")



        frame = cv.imread(frame_path)

        if frame is None:
            raise RuntimeError(f"Could not load frame: {frame_path}")



        # manually select subject bbox
        x, y, w, h = cv.selectROI(
            "Selecione o mero",
            frame,
            showCrosshair=True,
            fromCenter=False,
        )

        cv.destroyAllWindows()

        box = np.array(
            [x, y, x + w, y + h],
            dtype=np.float32
        )

        print("Bounding box:", box)

        metadata: VisMetadata = {
            "video_id": vid_id,
            "initial_frame": frame_idx,
            "initial_bbox": box.tolist(),
            "objects": {}
        }


        obj_id = 1

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=box,
        )



        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            frame_path = path_to_frame(vid_id, out_frame_idx)
            frame = cv.imread(frame_path)

            if frame is None:
                print(f"Could not load frame: {frame_path}")

                continue

            visualization = frame.copy()

            # assuming a single subject being tracked per frame
            for i, current_obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0).cpu().numpy()
                mask = np.squeeze(mask)

                if mask.shape != frame.shape[:2]:
                    print(f"Mask shape {mask.shape} does not match frame shape {frame.shape[:2]} for frame {out_frame_idx}")

                    continue


                ys, xs = np.where(mask)

                if len(xs) == 0 or len(ys) == 0:
                    continue

                x1 = int(xs.min())
                y1 = int(ys.min())
                x2 = int(xs.max())
                y2 = int(ys.max())

                bbox = [x1, y1, x2, y2]

                mask_area = int(mask.sum())

                obj_key = str(int(current_obj_id))

                if obj_key not in metadata["objects"]:
                    metadata["objects"][obj_key] = {
                        "frames": {}
                    }

                metadata["objects"][obj_key]["frames"][str(out_frame_idx)] = {
                    "bbox": bbox,
                    "mask_area": mask_area
                }


                overlay_color = np.array([0, 0, 255], dtype=np.uint8)

                visualization[mask] = (
                    visualization[mask].astype(np.float32) * 0.5
                    + overlay_color.astype(np.float32) * 0.5
                ).astype(np.uint8)

            output_path = path_to_vis(vid_id, out_frame_idx)
            cv.imwrite(output_path, visualization)

            print(f"Frame {out_frame_idx} written to {output_path}")

        persist_vis_metadata(vid_id, metadata)
