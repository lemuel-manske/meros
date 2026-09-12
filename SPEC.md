# Meros — Goliath Grouper Individual Re-Identification

## Project overview

**Meros** is a research collaboration between the **Regional University of Blumenau (FURB)** and the **Meros Institute of Brazil**.

The project aims to develop a computer-vision method capable of detecting and individually identifying **Goliath Groupers (*Epinephelus itajara*)** in underwater videos captured along the Brazilian coast.

The objective goes beyond determining whether a Goliath Grouper is present in a video. The system should determine **which individual fish is being observed**, allowing observations of the same animal from different videos, locations, and dates to be associated with one another.

Reliable individual identification would make it possible to study the displacement of Goliath Groupers along the Brazilian coast and, in particular, investigate **movement, site fidelity, and migration patterns associated with breeding aggregation areas**.

## Research problem

Given an underwater video containing one or more Goliath Groupers, the system should:

1. **Detect** Goliath Groupers in individual video frames.
2. **Track** each visible fish through the video.
3. Extract visual information that is useful for distinguishing individuals.
4. Combine information from multiple frames belonging to the same fish track.
5. Determine whether the observed fish corresponds to an individual previously recorded in the dataset.
6. When possible, assign the observation to the correct known individual.
7. Recognize when there is insufficient evidence to confidently identify the fish, or when the fish may represent an **individual not previously observed**.

The core research problem can therefore be formulated as **video-based individual re-identification (Re-ID) of Goliath Groupers under challenging underwater conditions**.

Importantly, the unit of identification should not necessarily be a single frame. A video may contain hundreds of observations of the same fish, and different frames may expose different anatomical regions or provide different levels of visual quality. The method should therefore exploit **temporal information across the video** whenever possible.

## Biological characteristics relevant to identification

Goliath Groupers are very large fish and exhibit several visual characteristics that may contain individual-specific information.

### Head patterns

The head contains numerous spots and irregular markings. Their spatial configuration may provide a useful biometric signature for distinguishing individuals.

### Body patterns

The body presents irregular combinations of darker and lighter regions, including bands, blotches, and other markings. These patterns may also contain information useful for individual identification.

### Color variation

Absolute body color should **not** be treated as a reliable identity feature.

An individual's apparent coloration can change because of biological color variation as well as environmental and imaging factors such as:

- depth;
- ambient illumination;
- artificial lighting;
- water turbidity;
- camera white balance;
- distance between the camera and the fish.

The identification method should therefore prioritize **spatial patterns, texture, morphology, and other features that remain relatively stable despite changes in color and brightness**.

Whether particular markings are sufficiently stable and unique to function as biometric identifiers is itself an empirical question that should be evaluated during the research.

## Dataset

The dataset consists primarily of short underwater videos recorded by **divers and swimmers along the Brazilian coast**, including observations made at breeding aggregation sites.

Videos generally contain approximately **100–500 frames**, with many sequences containing around **200 frames**.

Because consecutive frames are highly correlated, however, the number of video frames should not be interpreted as the number of independent observations. A 200-frame video may contain many nearly identical views of the same individual.

The effective amount of identity information therefore depends more strongly on:

- number of known individuals;
- number of independent encounters with each individual;
- number of locations;
- number of recording dates;
- viewpoint diversity;
- visibility conditions;
- quality of the visible anatomical patterns.

## Dataset characteristics

### Camera motion

The cameras are operated by divers or swimmers rather than mounted in a fixed position. Consequently:

- camera orientation changes continuously;
- distance to the fish varies;
- framing changes rapidly;
- motion blur can occur;
- the fish may enter or leave the image boundaries.

The system cannot assume a stable background or camera geometry.

### Viewpoint variation

Fish may be observed from substantially different angles, including:

- frontal views;
- left and right lateral views;
- oblique views;
- partial rear views.

The same identifying pattern may therefore be visible in one observation and completely absent in another.

This makes individual recognition inherently **viewpoint dependent**.

### Turbidity and visibility

Breeding areas may contain highly turbid water. Suspended particles reduce contrast and obscure fine-scale skin patterns.

Visibility may also vary considerably within a single recording as the distance between the camera and fish changes.

### Illumination and color distortion

Underwater lighting produces significant variation in:

- brightness;
- contrast;
- hue;
- color saturation.

The appearance of the same body region can therefore vary substantially depending on illumination, water conditions, and camera-to-fish distance.

This further supports avoiding direct dependence on absolute color for identification.

### Focus and image sharpness

Because of turbidity, movement, and limitations of underwater cameras, the fish may appear at different levels of sharpness.

Some frames expose detailed skin markings, while neighboring frames may contain little usable texture.

### Scale variation

Fish can occupy a large portion of the image when close to the camera or appear substantially smaller when farther away.

The identification method must therefore operate across significant differences in apparent scale.

### Occlusion

Fish may be:

- partially outside the frame;
- obscured by other Goliath Groupers;
- obscured by other marine organisms;
- obscured by underwater structures or environmental elements.

An identification system must therefore tolerate incomplete observations.

### Multiple individuals

Some recordings contain multiple Goliath Groupers at the same time.

Consequently, detection and tracking are important preprocessing stages. Identity evidence from different animals must not accidentally be aggregated into the same representation.

## Key research challenges

The project involves several related computer-vision challenges.

### 1. Fish detection

Locate every visible Goliath Grouper within each frame.

The detector must remain robust under:

- poor visibility;
- partial occlusion;
- variable scale;
- unusual viewing angles;
- motion blur.

### 2. Multi-object tracking

Associate detections belonging to the same fish across consecutive frames.

Tracking provides two important benefits:

- it prevents every frame from being treated as an independent observation;
- it produces a **tracklet** containing multiple views of the same candidate individual.

### 3. Anatomical region localization

The complete fish may not always provide the best representation for identification.

Potentially useful regions include:

- head;
- cheek or operculum region;
- lateral body;
- dorsal region;
- tail;
- other morphological structures.

The system may therefore benefit from identifying anatomical regions before extracting identity features.

### 4. Individual feature representation

The system must learn or construct an embedding in which observations of the **same fish are close together**, while observations of **different fish are separated**.

The representation should ideally be robust to:

- viewpoint;
- scale;
- brightness;
- color changes;
- turbidity;
- partial occlusion;
- moderate blur.

At the same time, it must preserve subtle individual-specific markings.

### 5. Multi-frame evidence aggregation

Not every frame provides equally useful information.

Rather than identifying a fish independently from every frame, the system should potentially estimate the quality of each observation and combine evidence across a track.

For example, one frame might provide a clear view of the head while another provides a better lateral body view.

This suggests a formulation such as:

**frames → fish detections → tracklet → quality/view assessment → identity representation → individual match**

rather than:

**frame → identity**

### 6. Cross-video re-identification

The central task is determining whether fish appearing in different recordings correspond to the same individual.

This is significantly more difficult than tracking within a single video because recordings may differ in:

- date;
- location;
- camera;
- illumination;
- water conditions;
- orientation;
- visible anatomical region.

### 7. Open-set identification

In real deployments, not every observed fish will already exist in the reference database.

The system should therefore distinguish between:

**Closed-set identification**

> Which known individual is this?

and:

**Open-set identification**

> Is this one of the known individuals, or is it a previously unseen fish?

This distinction is especially important if the system will eventually be used for ecological monitoring rather than only retrospective classification of a fixed dataset.

## Proposed system abstraction

A useful high-level representation of the pipeline is:

**Underwater video**

→ **Goliath Grouper detection**

→ **multi-object tracking**

→ **fish tracklets**

→ **frame/view quality assessment**

→ **anatomical-pattern feature extraction**

→ **multi-frame identity embedding**

→ **similarity search against known individuals**

→ **known individual / unknown individual decision**

→ **observation database**

→ **movement and migration analysis**

## Dataset annotation requirements

To properly investigate the problem, the dataset should eventually contain annotations at several levels.

At minimum:

- video identifier;
- recording date;
- recording location;
- known fish identity, when available;
- bounding box or segmentation of each visible grouper;
- association between detections belonging to the same fish within a video.

Additional annotations could substantially help the research:

- left/right/front/rear viewpoint;
- head visibility;
- body visibility;
- degree of occlusion;
- image quality;
- blur level;
- visibility or turbidity level;
- confidence of the human-provided identity.

These labels could later be automated, but they would make controlled experiments considerably easier.

## Evaluation protocol

Evaluation must be designed carefully because video frames are highly correlated.

Randomly splitting frames into training and testing sets would produce severe **data leakage**: nearly identical frames of the same encounter could appear in both sets and result in unrealistically high performance.

Dataset partitions should therefore be created at the level of **independent encounters or videos**, and preferably also account for recording date and location.

For individual re-identification, relevant metrics include:

- **Top-1 identification accuracy**;
- **Top-k identification accuracy**;
- **mean Average Precision (mAP)** for retrieval;
- **ROC / verification performance** for deciding whether two observations correspond to the same individual;
- **false-match rate**;
- **false-non-match rate**;
- performance of **unknown-individual rejection** for open-set experiments.

Results should additionally be stratified according to difficult conditions such as:

- viewpoint;
- visibility;
- blur;
- fish size in the image;
- occlusion;
- anatomical region visible.

## Research questions

The project should investigate questions such as:

1. **Can Goliath Groupers be reliably identified from natural skin patterns in unconstrained underwater video?**

2. **Which anatomical regions provide the strongest individual identity signal?**

3. **How stable are those patterns across different recording dates, environmental conditions, and viewpoints?**

4. **How much does video-level aggregation improve identification compared with single-frame recognition?**

5. **How should frames of different quality and viewpoints be combined?**

6. **How well can an identification model generalize to encounters recorded at different locations or times?**

7. **Can the system reliably distinguish known individuals from previously unseen individuals?**

8. **What minimum image quality or visible anatomical area is required before an identification should be considered reliable?**

## Long-term objective

The final goal is not merely to classify images of Goliath Groupers, but to construct a system capable of transforming opportunistically collected underwater video into **individual-level ecological observations**.

For each sufficiently reliable observation, the system should eventually produce information of the form:

**individual identity + location + date/time + identification confidence**

Repeated observations could then be used to reconstruct movement histories and investigate migration, site fidelity, and breeding aggregation behavior of *Epinephelus itajara* along the Brazilian coast.
