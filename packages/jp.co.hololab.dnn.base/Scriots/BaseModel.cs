using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using UnityEngine;
using Unity.Sentis;
using Unity.Sentis.ONNX;

namespace HoloLab.DNN.Base
{
    /// <summary>
    /// dnn base model for image recognition tasks
    /// </summary>
    public class BaseModel : IDisposable
    {
        protected Model runtime_model = null;
        protected IWorker worker = null;
        protected Ops ops = null;
        private Material pre_process = null;
        private RenderTexture render_texture = null;

        /// <summary>
        /// create base model from onnx file
        /// </summary>
        /// <param name="file_path">model file path</param>
        /// <param name="backend_type">backend type for inference engine</param>
        public BaseModel(string file_path, BackendType backend_type = BackendType.GPUCompute)
        {
            var is_optimize = true;
            var converter = new ONNXModelConverter(is_optimize, file_path);
            runtime_model = converter.Convert();
            Initialize(backend_type);
        }

        /// <summary>
        /// create base model from model asset
        /// </summary>
        /// <param name="model_asset">model asset</param>
        /// <param name="backend_type">backend type for inference engine</param>
        public BaseModel(ModelAsset model_asset, BackendType backend_type = BackendType.GPUCompute)
        {
            runtime_model = ModelLoader.Load(model_asset);
            Initialize(backend_type);
        }

        /// <summary>
        /// dispose base model
        /// </summary>
        public void Dispose()
        {
            worker?.Dispose();
            worker = null;

            ops?.Dispose();
            ops = null;

            if (render_texture != null)
            {
                RenderTexture.ReleaseTemporary(render_texture);
                render_texture = null;
            }
        }

        private void Initialize(BackendType backend_type)
        {
            worker = WorkerFactory.CreateWorker(backend_type, runtime_model);
            ops = WorkerFactory.CreateOps(backend_type, null);
            var input_width = runtime_model.inputs[0].shape[3].value;
            var input_height = runtime_model.inputs[0].shape[2].value;
            render_texture = RenderTexture.GetTemporary(input_width, input_height, 0, RenderTextureFormat.ARGBHalf);
            pre_process = new Material(Shader.Find("BaseModel/PreProcess"));
            pre_process.SetVector("_Mean", new Vector4(0.0f, 0.0f, 0.0f));
            pre_process.SetVector("_Std", new Vector4(1.0f, 1.0f, 1.0f));
            pre_process.SetFloat("_Max", 1.0f);
        }

        /// <summary>
        /// get input layer shape
        /// </summary>
        /// <returns>input layer shape (width, height, and channels)</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public (int width, int height, int channels) GetInputShape()
        {
            var input_width = runtime_model.inputs[0].shape[3].value;
            var input_height = runtime_model.inputs[0].shape[2].value;
            var input_channels = runtime_model.inputs[0].shape[1].value;
            return (input_width, input_height, input_channels);
        }

        /// <summary>
        /// get output layers shape
        /// </summary>
        /// <returns>the output layers name with shape</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Dictionary<string, TensorShape> GetOutputShapes()
        {
            var input_tensors = new Dictionary<string, Tensor>(runtime_model.inputs.Count);
            runtime_model.inputs.ForEach(input =>
            {
                var input_shape = input.shape.ToTensorShape();
                var input_tensor = TensorFloat.Zeros(input_shape);
                input_tensors[input.name] = input_tensor;
            });

            worker.Execute(input_tensors);

            var output_shapes = new Dictionary<string, TensorShape>(runtime_model.outputs.Count);
            runtime_model.outputs.ForEach(output =>
                output_shapes[output] = worker.PeekOutput(output).shape
            );

            return output_shapes;
        }

        /// <summary>
        /// set backend type for inference engine
        /// </summary>
        /// <param name="backend_type">backend type for inference engine</param>
        public void SetBackendType(BackendType backend_type)
        {
            worker?.Dispose();
            worker = WorkerFactory.CreateWorker(backend_type, runtime_model);

            ops?.Dispose();
            ops = WorkerFactory.CreateOps(backend_type, null);
        }

        /// <summary>
        /// set input layer shape
        /// </summary>
        /// <remarks>
        /// NOTE: usually, this method is not used because the input size is automatically set from the shape of the model's input layer.
        /// </remarks>
        /// <param name="widht">width</param>
        /// <param name="height">height</param>
        public void SetInputSize(int widht, int height)
        {
            if (render_texture.width == widht && render_texture.height == height)
            {
                return;
            }
            RenderTexture.ReleaseTemporary(render_texture);
            render_texture = RenderTexture.GetTemporary(widht, height, 0, RenderTextureFormat.ARGBHalf);
        }

        /// <summary>
        /// set value of input tensor normalization in pre process (mean)
        /// </summary>
        /// <remarks>
        /// default mean value is (1.0f, 1.0f, 1.0f), so it is not normalized.
        /// </remarks>
        /// <param name="mean">mean value</param>
        public void SetInputMean(Vector4 mean)
        {
            pre_process.SetVector("_Mean", mean);
        }

        /// <summary>
        /// set value of input tensor normalization in pre process (std)
        /// </summary>
        /// <remarks>
        /// default std value is (0.0f, 0.0f, 0.0f), so it is not normalized.
        /// </remarks>
        /// <param name="std">std value</param>
        public void SetInputStd(Vector4 std)
        {
            pre_process.SetVector("_Std", std);
        }

        /// <summary>
        /// set value of input tensor normalization in pre process (max)
        /// </summary>
        /// <remarks>
        /// default max value is 1.0f, so it is not normalized.
        /// if model requires input values in the range [0.0f-255.0f], you can set max value to 255.0f.
        /// </remarks>
        /// <param name="max">max value</param>
        public void SetInputMax(float max)
        {
            pre_process.SetFloat("_Max", max);
        }

        /// <summary>
        /// run predict and get output tensors
        /// </summary>
        /// <param name="image">input image</param>
        /// <returns>output layers name with tensor</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Dictionary<string, Tensor> Predict(Texture2D image)
        {
            (var input_width, var input_height, var input_channels) = GetInputShape();
            var input_texture = PreProcess(image);
            var input_tensor = TextureConverter.ToTensor(input_texture, input_width, input_height, input_channels);

            worker.Execute(input_tensor);

            var output_tensors = new Dictionary<string, Tensor>(runtime_model.outputs.Count);
            runtime_model.outputs.ForEach(output => {
                var output_tensor = worker.PeekOutput(output);
                output_tensor.TakeOwnership();
                output_tensors[output] = output_tensor;
            });

            input_tensor.Dispose();
            MonoBehaviour.Destroy(input_texture);

            return output_tensors;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Texture PreProcess(Texture2D image)
        {
            RenderTexture.active = render_texture;
            Graphics.Blit(image, render_texture, pre_process);

            var result = new Texture2D(render_texture.width, render_texture.height, TextureFormat.RGBAHalf, false);
            result.ReadPixels(new Rect(0, 0, render_texture.width, render_texture.height), 0, 0);
            result.Apply();

            return result;
        }
    }
}
