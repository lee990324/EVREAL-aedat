import numpy as np
from dv import AedatFile
import sys
import os
import json
import argparse
import glob
import aedat
import cv2

def aedat_to_npy(aedat_path, output_pth):
    metadata_path = os.path.join(output_pth, 'metadata.json')
    events_ts_path = os.path.join(output_pth, 'events_ts.npy')
    events_xy_path = os.path.join(output_pth, 'events_xy.npy')
    events_p_path = os.path.join(output_pth, 'events_p.npy')
    images_path = os.path.join(output_pth, 'images.npy')
    images_ts_path = os.path.join(output_pth, 'images_ts.npy')
    image_event_indices_path = os.path.join(output_pth, 'image_event_indices.npy')
    sensor_size = None
    image_list, image_ts_list, image_event_indices_list = [], [], []

    with AedatFile(aedat_path) as f:
        events = np.hstack([packet for packet in f['events'].numpy()])

    # Extract the columns
    xs = events['x']
    ys = events['y']
    timestamps = events['timestamp']
    ps = events['polarity']

    ts = []
    for i in range(len(timestamps)):
        temp = str(timestamps[i])
        ts.append(float(temp[:10] + "." + temp[10:])) # 첫 번째 열의 숫자에 대해 10번째 자리에 '.'을 삽입합니다.

    decoder = aedat.Decoder(aedat_path)
    index = 0
    for packet in decoder:
        if "frame" in packet:
            image = cv2.cvtColor(packet["frame"]["pixels"], cv2.COLOR_BGR2GRAY)
            image_list.append(image)
            temp2 = str(packet["frame"]["t"])
            image_ts_list.append(float(temp2[:10] + "." + temp2[10:]))

            if sensor_size is None:
                sensor_size = image.shape[:2]
            index += 1

    events_ts = np.array(ts)
    events_xy = np.array([xs, ys]).transpose()
    events_p = np.array(ps)

    images = np.stack(image_list)
    images_ts = np.stack(image_ts_list)

    images = np.expand_dims(images, axis=-1)
    images_ts = np.expand_dims(images_ts, axis=1)

    image_event_indices = np.searchsorted(events_ts, images_ts, 'right') - 1
    image_event_indices = np.clip(image_event_indices, 0, len(events_ts) - 1)

    np.save(events_ts_path, events_ts, allow_pickle=False, fix_imports=False)
    np.save(events_xy_path, events_xy, allow_pickle=False, fix_imports=False)
    np.save(events_p_path, events_p, allow_pickle=False, fix_imports=False)

    np.save(images_path, images, allow_pickle=False, fix_imports=False)
    np.save(images_ts_path, images_ts, allow_pickle=False, fix_imports=False)
    np.save(image_event_indices_path, image_event_indices, allow_pickle=False, fix_imports=False)

    metadata = {"sensor_resolution": sensor_size}
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

if __name__ == "__main__":
    """ Tool for converting rosbag events and images to numpy format. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Directory of ROS bags")
    parser.add_argument("--event_topic", default="/dvs/events", help="Event topic")
    parser.add_argument("--image_topic", default="/dvs/image_raw", help="Image topic")
    parser.add_argument("--remove", help="Remove rosbags after conversion", action="store_true")
    args = parser.parse_args()
    aedat_paths = sorted(glob.glob(os.path.join(args.path, "*.aedat4")))
    for path in aedat_paths:
        print("Processing {}".format(path))
        output_pth = os.path.splitext(path)[0]
        os.makedirs(output_pth, exist_ok=True)
        try:
            aedat_to_npy(path, output_pth)
        except Exception as e:
            print("Failed to convert {}".format(path))
            print(e)
            continue
        if args.remove:
            os.remove(path)
