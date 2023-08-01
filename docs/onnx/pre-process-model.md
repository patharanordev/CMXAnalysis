# **Built-in model pre-processing with ONNX**

## **Add pre-processing to the model**

Instead of having to write the input image resizing and other manipulations in Kotlin, the pre- and post-processing steps can be added to the model using operators available in newer versions of the ONNX runtime (eg. ORT 1.14/opset 18) or in the `onnxruntime_extensions` library.

My analyzer is a great demonstration of this feature as both the input and output are images, so including the processing steps in the model greatly reduces platform-specific code.

The steps to add the pre- and post-processing to the model are in the [Prepare the model](https://onnxruntime.ai/docs/tutorials/mobile/superres.html#prepare-the-model) section of the sample documentation. The python script that creates the updated model can be viewed at [superresolution_e2e.py](https://raw.githubusercontent.com/microsoft/onnxruntime-extensions/main/tutorials/superresolution_e2e.py).

> The **superresolution_e2e** script is explained step-by-step on [pytorch.org](https://pytorch.org/)

When you run the script as instructed, it produces two models in ONNX format – the basic **pytorch_superresolution.onnx** model and another version that includes additional processing **pytorch_superresolution_with_pre_and_post_proceessing.onnx**. The second model, including the processing instructions, can be called in an Android app with fewer lines of code.

### **Example output**

Original NN:

![original-nn](../../assets/ori-nn-graph.png)

Pre-processing NN:

You will see `DecodeImage` node before original NN.

![original-nn](../../assets/pre-process-nn-graph.png)

## **References**

- [Add pre-processing to the model](https://devblogs.microsoft.com/surface-duo/onnx-machine-learning-4/).
