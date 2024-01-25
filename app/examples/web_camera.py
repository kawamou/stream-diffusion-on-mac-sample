import time
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image

from ..stream_diffusion import StreamDiffusion

SD_SIDE_LENGTH = 512


def main():
    stream = StreamDiffusion(prompt="cartoon, deepfake").stream

    cap = cv2.VideoCapture(0)

    img_dst = Image.new("RGB", (SD_SIDE_LENGTH * 2, SD_SIDE_LENGTH))

    while True:
        try:
            _, frame = cap.read()

            init_img = crop_center(Image.fromarray(frame), SD_SIDE_LENGTH * 2, SD_SIDE_LENGTH * 2).resize(
                (SD_SIDE_LENGTH, SD_SIDE_LENGTH), Image.NEAREST
            )

            image_tensor = stream.preprocess_image(init_img)

            output_image, fps = timeit(stream)(image=image_tensor)

            if isinstance(output_image, Image.Image):
                img_dst.paste(init_img, (0, 0))
                img_dst.paste(output_image, (SD_SIDE_LENGTH, 0))

                cv2.imshow("{} fps".format(fps), np.array(img_dst))
                cv2.waitKey(1)

        except KeyboardInterrupt:
            break

    cap.release()
    cv2.destroyAllWindows()


def timeit(func: Callable[..., Any]):
    def wrapper(*args, **kwargs) -> tuple[Image.Image, str]:
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start
        return result, f"{1 / elapsed_time}"

    return wrapper


def crop_center(pil_img: Image.Image, crop_width: int, crop_height: int):
    img_width, img_height = pil_img.size
    return pil_img.crop(
        (
            (img_width - crop_width) // 2,
            (img_height - crop_height) // 2,
            (img_width + crop_width) // 2,
            (img_height + crop_height) // 2,
        )
    )


if __name__ == "__main__":
    main()
