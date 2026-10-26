import cv2 as cv
import matplotlib.pyplot as plt

from src.meros.sandbox.sources import sources

if __name__ == "__main__":
    path = sources[0]

    img = cv.imread(path)
    assert img is not None

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT.create()

    keypoints, descriptors = sift.detectAndCompute(gray, None)

    r = cv.drawKeypoints(
        gray, keypoints, gray, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    plt.figure(figsize=(10, 6))
    plt.imshow(cv.cvtColor(r, cv.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
