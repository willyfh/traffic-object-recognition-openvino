import argparse
import cv2
import ast
import numpy as np
from inference import Network

INPUT_STREAM = "driving_jakarta.mp4"
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

semantic_label_colors = np.array([
    (192, 0, 128), (192, 0, 192), (0, 0, 0), (0, 0, 128), (128, 128,128),
    (0, 128,128), (128, 128, 0), (64, 128, 128), (0, 128, 0), (192, 0, 0),
    (64, 128, 0), (192, 128, 0), (64, 0, 128), (128, 0, 0), (64, 128, 128),
    (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0)
])

road_label_colors = np.array([
    (0, 0, 0),(128, 0, 0), (0, 0, 128), (0, 128, 0)
])

bounding_box_color = (255, 0, 0)

n_semantic_class = 20
n_road_class = 4

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    t_desc = "Model type. (VD , SS, or RS)" # VD: Vehicle Detection, SS: Semantic Segmentation, RS: Road Segmentation
    ct_desc = "The confidence threshold to use with the bounding boxes"

    # Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # Create the arguments
    required.add_argument("-m", help=m_desc, required=True)
    required.add_argument("-t", help=t_desc, required=True)
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-ct", help=ct_desc, default=0.5)
    
    args = parser.parse_args()
    args.ct = float(args.ct)
    return args

def preprocess(frame, net_input_shape):
    '''preprocess the given frame'''
    p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
    p_frame = p_frame.transpose((2,0,1))
    p_frame = p_frame.reshape(1, *p_frame.shape)
    return p_frame


def draw_boxes(frame, output, width, height, color, confidence_threshold):
    '''draw bounding boxes onto the frame'''
    for box in output[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= confidence_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)
    return frame


def segment_pixels(model_type, frame, output, width, height, n_class, label_colors):
    '''segment the frame based on the model output'''
    if model_type == 'SS':
        output = output[0][0]
    elif model_type == 'RS':
        output = np.argmax(output[0],axis=0).astype('float32')
    
    output = cv2.resize(output, (width, height))

    b = np.zeros_like(output).astype(np.uint8)
    g = np.zeros_like(output).astype(np.uint8)
    r = np.zeros_like(output).astype(np.uint8)

    for l in range(0, n_class):
        idxs = output == l
        b[idxs] = label_colors[l, 0]
        g[idxs] = label_colors[l, 1]
        r[idxs] = label_colors[l, 2]

    bgr = np.stack([b, g, r], axis=2)

    o_frame = ((0.4 * frame) + (0.6 * bgr)).astype("uint8")

    if model_type == 'RS':
        idxs = output == 0
        o_frame[idxs] = frame[idxs]
            
    return o_frame

def infer_on_video(args):
    '''inference function'''
    # Initialize the Inference Engine
    plugin = Network()
    
    # Load the network model into the IE
    plugin.load_model(args.m, args.d, CPU_EXTENSION)
    net_input_shape = plugin.get_input_shape()
    
    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Create a video writer for the output video
    # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
    # on Mac, and `0x00000021` on Linux
    out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (width,height))
    
    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        # Pre-process the frame
        p_frame = preprocess(frame, net_input_shape)
        
        # Perform inference on the frame
        plugin.async_inference(p_frame)
        
        if plugin.wait() == 0:
            # Get the output of inference
            output = plugin.extract_output()
            
            # Post-process the output
            if args.t == 'VD': # vehicle detection
                o_frame = draw_boxes(frame, output, width, height, bounding_box_color, args.ct)
            elif args.t == 'RS': # road segmentation
                n_class = n_road_class
                label_colors = road_label_colors
                o_frame = segment_pixels(args.t, frame, output, width, height, n_class, label_colors)
            elif args.t == 'SS': # semantic segmentation
                n_class = n_semantic_class
                label_colors = semantic_label_colors
                o_frame = segment_pixels(args.t, frame, output, width, height, n_class, label_colors)
            else:
                print('type %s is not defined' % args.t)
                exit(1)
            
            # Write out the frame
            out.write(o_frame)

        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the out writer, capture, and destroy any OpenCV windows
    out.release()
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = get_args()
    infer_on_video(args)


if __name__ == "__main__":
    main()
