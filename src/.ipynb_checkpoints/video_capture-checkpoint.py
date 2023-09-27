

#%%
import cv2
import argparse

#% Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', default=0, type=int, help='Camera index.')
parser.add_argument('--verbose', help='increase output verbosity', action='store_true')
args = parser.parse_args()
index = args.i

def show_cap_properties(cap):

    # showing values of the properties
    print(f'CAP_PROP_FRAME_WIDTH: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}')
    print(f'CAP_PROP_FRAME_HEIGHT : {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}')
    print(f'CAP_PROP_FPS : {cap.get(cv2.CAP_PROP_FPS)}')
    print(f'CAP_PROP_POS_MSEC : {cap.get(cv2.CAP_PROP_POS_MSEC)}')
    print(f'CAP_PROP_FRAME_COUNT  : {cap.get(cv2.CAP_PROP_FRAME_COUNT)}')
    print(f'CAP_PROP_BRIGHTNESS : {cap.get(cv2.CAP_PROP_BRIGHTNESS)}')
    print(f'CAP_PROP_CONTRAST : {cap.get(cv2.CAP_PROP_CONTRAST)}')
    print(f'CAP_PROP_SATURATION : {cap.get(cv2.CAP_PROP_SATURATION)}')
    print(f'CAP_PROP_HUE : {cap.get(cv2.CAP_PROP_HUE)}')
    print(f'CAP_PROP_GAIN  : {cap.get(cv2.CAP_PROP_GAIN)}')
    print(f'CAP_PROP_CONVERT_RGB : {cap.get(cv2.CAP_PROP_CONVERT_RGB)}')

cap = cv2.VideoCapture(index)

try:

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

    if args.verbose:
        show_cap_properties(cap)


    while cap.isOpened():

        ret, frame = cap.read()
        if ret:
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except cv2.error as error:
    print('[Error]: {}'.format(error))

cap.release()
cv2.destroyAllWindows()