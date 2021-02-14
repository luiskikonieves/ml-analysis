import cv2
import numpy as np
import youtube_dl
import analyzer
import sort

from PIL import Image


def analyze(video_url):
    """
    Downloads a video from YouTube and runs it via a torchvision model, displaying the results.

    :param string video_url: YouTube URL
    """
    # create youtube-dl object
    ydl = youtube_dl.YoutubeDL({})

    # set video url, extract video information
    info_dict = ydl.extract_info(video_url, download=False)
    # get video formats available
    formats = info_dict.get('formats', None)
    img_size = 300

    for f in formats:
        # TODO: Add this as a user CLI param?
        if f.get('format_note', None) == '720p':

            # get the video url and open it with opencv
            url = f.get('url', None)
            cap = cv2.VideoCapture(url)

            # check if url was opened, otherwise bail
            if not cap.isOpened():
                print('Couldn\'t open')
                exit(0)

            while True:
                # read frame
                ret, frame = cap.read()

                # check if frame is empty and bail if it is
                if not ret:
                    break
                # Run it through the model
                prediction = analyzer.imagenet(frame)

                pilimg = Image.fromarray(frame)
                img = np.array(pilimg)
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128),
                          (128, 0, 128), (128, 128, 0), (0, 128, 128)]
                pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
                pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
                unpad_h = img_size - pad_y
                unpad_w = img_size - pad_x

                mot_tracker = sort.Sort()
                if prediction is not None:
                    tracked_objects = mot_tracker.update(prediction.cpu())
                    unique_labels = prediction[:, -1].cpu().unique()
                    n_cls_preds = len(unique_labels)

                    for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                        box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                        box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                        y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                        x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                        color = colors[int(obj_id) % len(colors)]
                        cls = classes[int(cls_pred)]
                        cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), color, 4)
                        cv2.rectangle(frame, (x1, y1 - 35), (x1 + len(cls) * 19 + 80, y1), color, -1)
                        cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 255), 3)

                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (50, 50)
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                # frame = cv2.putText(frame, prediction, org, font, fontScale=1, color=(255, 0, 0), thickness=2)
                # display frame
                cv2.imshow('Video', frame)

                # Bail on ESC key input
                if cv2.waitKey(30) & 0xFF == 27:
                    break

            # Release resources
            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    video_url = 'https://www.youtube.com/watch?v=WluMkc3OtoM&ab_channel=4KUrbanLife'
    analyze(video_url)
