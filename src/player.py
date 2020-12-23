import cv2
import numpy as np
import youtube_dl
import analyzer

def analyze():
    video_url = 'https://www.youtube.com/watch?v=WluMkc3OtoM&ab_channel=4KUrbanLife'

    # create youtube-dl object
    ydl = youtube_dl.YoutubeDL({})

    # set video url, extract video information
    info_dict = ydl.extract_info(video_url, download=False)
    # get video formats available
    formats = info_dict.get('formats',None)

    for f in formats:
        # TODO: Add this as a user CLI param?
        if f.get('format_note',None) == '720p':

            #get the video url
            url = f.get('url',None)
            # open url with opencv
            cap = cv2.VideoCapture(url)

            # check if url was opened
            if not cap.isOpened():
                print('Couldn\'t open')
                exit(0)

            while True:
                # read frame
                ret, frame = cap.read()

                # check if frame is empty
                if not ret:
                    break
                # Run it through the model
                prediction = analyzer.infer(frame)

                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (50, 50)
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                frame = cv2.putText(frame, prediction, org, font, fontScale=1, color=(255, 0, 0), thickness=2)
                # display frame
                cv2.imshow('Video', frame)

                # Bail on ESC key input
                if cv2.waitKey(30)&0xFF == 27:
                    break

            # Release resources
            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    analyze()