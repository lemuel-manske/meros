# Related research

- https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cvi2.12337
- https://link.springer.com/article/10.1007/s11263-024-02071-1
- https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2022.1093623/full
- https://www.mdpi.com/1424-8220/21/19/6392
- https://royalsocietypublishing.org/rsos/article/8/7/201768/96496/Re-identification-of-individuals-from-images-using
- https://pmc.ncbi.nlm.nih.gov/articles/PMC8376999/
- https://www.mdpi.com/2076-3417/12/11/5401

## YOLO and individual identification

- [Wang et al. (2022), A Real-Time Individual Identification Method for Swimming Fish Based on Improved YOLOv5](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4044575).
  Proposes YOLOv5 with CBAM attention and BiFPN for individual fish identification.
  SSRN preprint; the abstract was accessible, but the dataset and evaluation protocol
  have not been verified here.
- [Gao et al. (2022), FIRN: A Novel Fish Individual Recognition Method with Accurate Detection and Attention Mechanism](https://doi.org/10.3390/electronics11213459).
  Uses improved YOLOv4 for detection and a separate ResNeXt50-based recognition
  network trained with ArcFace loss. The dataset contains 8,000 images across four
  species, with 30–50 individuals per species.
- [Zhao et al. (2025), Fine-Grained Fish Individual Recognition in Underwater Environments Using Global Detail Enhancement and Keypoint Region Fusion](https://doi.org/10.3390/fishes10030102).
  Uses YOLOv8 to detect fish and nine anatomical keypoints, then combines global
  and local features in a separate recognition network. Reports 95.3% Rank-1 and
  89.7% mAP on 5,000 images of 29 largemouth bass from controlled tanks. These
  results do not establish performance on wild groupers across independent encounters.
- [Liang et al. (2026), Occlusion-Robust Fish Individual Identification: A Multi-Model Video Recognition Framework for Complex Underwater Environments](https://www.sciencedirect.com/science/article/pii/S0168169926010847).
  Combines YOLOv9 detection, SORT tracking, and fused CosPlace/DOLG representations
  for identity retrieval. Recognition is performed by the complete pipeline,
  rather than YOLO alone.
- [Jahn et al. (2025), Individual Behavior Tracking of Heifers by Using Object Detection Algorithm YOLOv4](https://doi.org/10.3389/fanim.2024.1499253).
  Assigns each of five heifers its own detection class, recognizing individuals
  from coat patterns. A direct example of using YOLO to predict known identities;
  it does not establish recognition of previously unseen animals.

## Identification across seasons and years

- [Shi et al. (2023), Aging Contrast: A Contrastive Learning Framework for Fish Re-identification Across Seasons and Years](https://doi.org/10.1007/978-981-99-8388-9_21).
  Uses contrastive learning and cosine similarity to recognize tagged masu salmon
  across seasons and years. Although it does not use YOLO, its temporal evaluation
  is particularly relevant to Meros.

## Supporting methods

- https://arxiv.org/html/2006.04250
