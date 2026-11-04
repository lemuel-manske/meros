# Meros - Goliath Grouper Individual Re-Identification

## Project overview

> Meros is the Portuguese name for Goliath Groupers (*Epinephelus itajara*)

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

#### Observed longitudinal variation in spot appearance

Two photographs from different encounters of the fish now identified as **Rogerio** show a local difference in the marked skin pattern near the eye: one photograph appears to contain a single larger dark spot, while the other shows two smaller adjacent marks in the corresponding region. The observer confirmed that both photographs depict the same individual.

This is an **observed difference in appearance across encounters**, not proof that one biological spot physically split into two. Camera viewpoint, image quality, illumination, and mark boundaries may contribute to the appearance; the biological mechanism has not been established.

### Body patterns

The body presents irregular combinations of darker and lighter regions, including bands, blotches, and other markings. These patterns may also contain information useful for individual identification.

It's known that those patterns can change their color over time; Some captures show a very dark body, while others show a lighter body with no visible markings.

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

The current working dataset contains 6 individuals, with 2 of them with multiple captures.

Those photos were taken on Fernando de Noronha Island, Brazil, in 2004–2013. The dataset is **not yet large enough to support a fully independent evaluation of identification performance**.

The dataset was labeled: individual name, head spots, spots correspondences between captures, mouths, eyes, etc.

## Dataset cearacteristics

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

## Key research challenges

### Open-set identification

In real deployments, not every observed fish will already exist in the reference database.

The system should therefore distinguish between:

**Closed-set identification**

> Which known individual is this?

and:

**Open-set identification**

> Is this one of the known individuals, or is it a previously unseen fish?

This distinction is especially important if the system will eventually be used for ecological monitoring rather than only retrospective classification of a fixed dataset.

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
