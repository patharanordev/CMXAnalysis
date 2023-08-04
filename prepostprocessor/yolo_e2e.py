# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy

import onnx
from pathlib import Path
import onnxruntime_extensions


def get_yolov8_model(onnx_model_name: str):
    # install yolov8
    from pip._internal import main as pipmain
    try:
        import ultralytics
    except ImportError:
        pipmain(['install', 'ultralytics'])
        import ultralytics
    pt_model = Path("yolov8n.pt")
    model = ultralytics.YOLO(str(pt_model))  # load a pretrained model
    success = model.export(format="onnx")  # export the model to ONNX format
    assert success, "Failed to export yolov8n.pt to onnx"
    import shutil
    shutil.move(pt_model.with_suffix('.onnx'), onnx_model_name)


def add_pre_post_processing_to_yolo(input_model_file: Path, output_model_file: Path):
    """Construct the pipeline for an end2end model with pre and post processing. 
    The final model can take raw image binary as inputs and output the result in raw image file.

    Args:
        input_model_file (Path): The onnx yolo model.
        output_model_file (Path): where to save the final onnx model.
    """
    if not Path(input_model_file).is_file():
        get_yolov8_model(input_model_file)

    # from onnxruntime_extensions.tools import add_pre_post_processing_to_model as add_ppp
    # add_ppp.yolo_detection(input_model_file, output_model_file, "jpg", onnx_opset=18)


    # Ref: 
    #  - add_pre_post_processing_to_model : https://github.com/microsoft/onnxruntime-extensions/blob/main/onnxruntime_extensions/tools/add_pre_post_processing_to_model.py#L255
    #  - vision : https://github.com/microsoft/onnxruntime-extensions/blob/911c2b23409409266c6a2a56d28f1f093d65d1c0/onnxruntime_extensions/tools/pre_post_processing/steps/vision.py#L629
    
    # NOTE: If you're working on this script install onnxruntime_extensions using `pip install -e .` from the repo root
    # and run with `python -m onnxruntime_extensions.tools.add_pre_post_processing_to_model`
    # Running directly will result in an error from a relative import.
    from onnxruntime_extensions.tools.pre_post_processing import pre_post_processor, step, steps, utils

    model_file = input_model_file
    output_file = output_model_file
    output_format = "jpg"
    onnx_opset = 18
    num_classes = 80
    input_shape: List[int] = None

    model = onnx.load(str(model_file.resolve(strict=True)))
    inputs = [utils.create_named_value("image", onnx.TensorProto.UINT8, ["num_bytes"])]

    model_input_shape = model.graph.input[0].type.tensor_type.shape
    model_output_shape = model.graph.output[0].type.tensor_type.shape

    # We will use the input_shape to create the model if provided by user.
    if input_shape is not None:
        assert len(input_shape) == 2, "The input_shape should be [h, w]."
        w_in = input_shape[1]
        h_in = input_shape[0]
    else:
        assert (model_input_shape.dim[-1].HasField("dim_value") and
                model_input_shape.dim[-2].HasField("dim_value")), "please provide input_shape in the command args."

        w_in = model_input_shape.dim[-1].dim_value
        h_in = model_input_shape.dim[-2].dim_value

    # Yolov5(v3,v7) has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
    # Yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
    # https://github.com/ultralytics/ultralytics/blob/e5cb35edfc3bbc9d7d7db8a6042778a751f0e39e/examples/YOLOv8-CPP-Inference/inference.cpp#L31-L33
    # We always want the box info to be the last dim for each of iteration.
    # For new variants like YoloV8, we need to add an transpose op to permute output back.
    need_transpose = False

    output_shape = [model_output_shape.dim[i].dim_value if model_output_shape.dim[i].HasField("dim_value") else -1
                    for i in [-2, -1]]
    if output_shape[0] != -1 and output_shape[1] != -1:
        need_transpose = output_shape[0] < output_shape[1] 
    else:
        assert len(model.graph.input) == 1, "Doesn't support adding pre and post-processing for multi-inputs model."
        try:
            import numpy as np
            import onnxruntime
        except ImportError:
            raise ImportError(
                """Please install onnxruntime and numpy to run this script. eg 'pip install onnxruntime numpy'.
Because we need to execute the model to determine the output shape in order to add the correct post-processing""")

        # Generate a random input to run the model and infer the output shape.
        session = onnxruntime.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        input_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[model.graph.input[0].type.tensor_type.elem_type]
        inp = {input_name: np.random.rand(1, 3, h_in, w_in).astype(dtype=input_type)}
        outputs = session.run(None,  inp)[0]
        assert len(outputs.shape) == 3 and outputs.shape[0] == 1, "shape of the first model output is not (1, n, m)"
        if outputs.shape[1] < outputs.shape[2]:
            need_transpose = True
        assert num_classes+4 == outputs.shape[2] or num_classes+5 == outputs.shape[2], \
            "The output shape is neither (1, num_boxes, num_classes+4(reg)) nor (1, num_boxes, num_classes+5(reg+obj))"

    pipeline = pre_post_processor.PrePostProcessor(inputs, onnx_opset)
    # precess steps are responsible for converting any jpg/png image to CHW BGR float32 tensor
    # jpg-->BGR(Image Tensor)-->Resize (scaled Image)-->LetterBox (Fix sized Image)-->(from HWC to)CHW-->float32-->1CHW
    pipeline.add_pre_processing(
        [
            steps.vision.ConvertImageToBGR(),  # jpg/png image to BGR in HWC layout
            # Resize an arbitrary sized image to a fixed size in not_larger policy
            steps.vision.Resize((h_in, w_in), policy='not_larger'),
            steps.vision.LetterBox(target_shape=(h_in, w_in)),  # padding or cropping the image to (h_in, w_in)
            steps.vision.ChannelsLastToChannelsFirst(),  # HWC to CHW
            steps.vision.ImageBytesToFloat(),  # Convert to float in range 0..1
            steps.general.Unsqueeze([0]),  # add batch, CHW --> 1CHW
        ]
    )
    # NMS and drawing boxes
    post_processing_steps = [
        steps.general.Squeeze([0]), # - Squeeze to remove batch dimension
        steps.vision.SplitOutBoxAndScore(num_classes=num_classes), # Separate bounding box and confidence outputs
        steps.vision.SelectBestBoundingBoxesByNMS(), # Apply NMS to suppress bounding boxes
        (
            steps.vision.ScaleBoundingBoxes(),  # Scale bounding box coords back to original image
            [
                # A connection from original image to ScaleBoundingBoxes
                # A connection from the resized image to ScaleBoundingBoxes
                # A connection from the LetterBoxed image to ScaleBoundingBoxes
                # We can use the three image to calculate the scale factor and offset.
                # With scale and offset, we can scale the bounding box back to the original image.
                utils.IoMapEntry("ConvertImageToBGR", producer_idx=0, consumer_idx=1),
                utils.IoMapEntry("Resize", producer_idx=0, consumer_idx=2),
                utils.IoMapEntry("LetterBox", producer_idx=0, consumer_idx=3),
            ],
        ),
        # Added debugger to access "scaled_box_out", 
        # it's the output from "ScaleBoundingBoxes" or boxes.
        step.Debug(),
        # DrawBoundingBoxes on the original image
        # Model imported from pytorch has CENTER_XYWH format
        # two mode for how to color box,
        #   1. colour_by_classes=True, (colour_by_classes), 2. colour_by_classes=False,(colour_by_confidence)
        (steps.vision.DrawBoundingBoxes(mode='CENTER_XYWH', num_classes=num_classes, colour_by_classes=True),
         [
            utils.IoMapEntry("ConvertImageToBGR", producer_idx=0, consumer_idx=0),
            utils.IoMapEntry("ScaleBoundingBoxes", producer_idx=0, consumer_idx=1),
        ]),
        # Encode to jpg/png
        steps.vision.ConvertBGRToImage(image_format=output_format),
    ]
    # transpose to (num_boxes, coor+conf) if needed
    if need_transpose:
        post_processing_steps.insert(1, steps.general.Transpose([1, 0]))

    pipeline.add_post_processing(post_processing_steps)

    new_model = pipeline.run(model)
    new_model = onnx.shape_inference.infer_shapes(new_model)

    # Modify output
    #graph = new_model.graph

    onnx.save_model(new_model, str(output_file.resolve()))


def test_inference(onnx_model_file:Path):
    import onnxruntime as ort
    import numpy as np

    providers = ['CPUExecutionProvider']
    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(onnxruntime_extensions.get_library_path())

    image = np.frombuffer(open('./test/data/ppp_vision/wolves.jpg', 'rb').read(), dtype=np.uint8)
    session = ort.InferenceSession(str(onnx_model_file), providers=providers, sess_options=session_options)

    inname = [i.name for i in session.get_inputs()]
    inp = {inname[0]: image}
    outputs = session.run(['image_out', 'scaled_box_out_next', 'scaled_box_out_debug'], inp)[0]
    open('./test/data/result.jpg', 'wb').write(outputs)

if __name__ == '__main__':
    print("checking the model...")
    onnx_model_name = Path("test/data/yolov8n.onnx")
    onnx_e2e_model_name = onnx_model_name.with_suffix(suffix=".with_pre_post_processing.onnx")
    add_pre_post_processing_to_yolo(onnx_model_name, onnx_e2e_model_name)
    test_inference(onnx_e2e_model_name)