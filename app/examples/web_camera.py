import time

import cv2
import numpy as np
from PIL import Image

from ..stream_diffusion import StreamDiffusion


def main():
    prev_time = time.time()

    stream = StreamDiffusion(prompt="1man, expressive, pop art style").stream

    cap = cv2.VideoCapture(0)

    img_dst = Image.new("RGB", (1024, 512))

    while True:
        ret, frame = cap.read()

        width, height = frame.shape[1], frame.shape[0]
        left = (width - 1024) // 2
        top = (height - 1024) // 2
        right = (width + 1024) // 2
        bottom = (height + 1024) // 2

        img_init = Image.fromarray(frame).crop((left, top, right, bottom)).resize((512, 512), Image.NEAREST)

        image_tensor = stream.preprocess_image(img_init)

        output_image = stream(image=image_tensor)

        if isinstance(output_image, Image.Image):
            img_dst.paste(img_init, (0, 0))
            img_dst.paste(output_image, (512, 0))

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            fps_str = "{} fps".format(str(fps))

            cv2.imshow(fps_str, np.array(img_dst))
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
