using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using UnityEngine;
using Unity.Sentis;
using Unity.Sentis.Quantization;

namespace HoloLab.DNN.Base
{
    /// <summary>
    /// dnn base model for image recognition tasks
    /// </summary>
    public class BaseModel : IDisposable
    {
        protected Model runtime_model = null;
        protected IWorker worker = null;
        protected IBackend backend = null;
        protected IEnumerator schedule = null;
        private Material pre_process = null;
        private RenderTexture render_texture = null;
        private bool is_predicting = false;
        private int layers_per_frame = 5;

        /// <summary>
        /// create base model from sentis file
        /// </summary>
        /// <param name="file_path">model file path</param>
        /// <param name="backend_type">backend type for inference engine</param>
        /// <param name="apply_quantize">apply float16 quantize</param>
        public BaseModel(string file_path, BackendType backend_type = BackendType.GPUCompute, bool apply_quantize = false)
        {
            runtime_model = ModelLoader.Load(file_path);
            Initialize(backend_type, apply_quantize);
        }

        /// <summary>
        /// create base model from stream
        /// </summary>
        /// <param name="stream">mdoel stream</param>
        /// <param name="backend_type">backend type for inference engine</param>
        /// <param name="apply_quantize">apply float16 quantize</param>
        public BaseModel(System.IO.Stream stream, BackendType backend_type = BackendType.GPUCompute, bool apply_quantize = false)
        {
            runtime_model = ModelLoader.Load(stream);
            Initialize(backend_type, apply_quantize);
        }

        /// <summary>
        /// create base model from model asset
        /// </summary>
        /// <param name="model_asset">model asset</param>
        /// <param name="backend_type">backend type for inference engine</param>
        /// <param name="apply_quantize">apply float16 quantize</param>
        public BaseModel(ModelAsset model_asset, BackendType backend_type = BackendType.GPUCompute, bool apply_quantize = true)
        {
            runtime_model = ModelLoader.Load(model_asset);
            Initialize(backend_type, apply_quantize);
        }

        /// <summary>
        /// dispose base model
        /// </summary>
        public void Dispose()
        {
            worker?.Dispose();
            worker = null;

            backend?.Dispose();
            backend = null;

            if (render_texture != null)
            {
                RenderTexture.ReleaseTemporary(render_texture);
                render_texture = null;
            }
        }

        private void Initialize(BackendType backend_type, bool apply_quantize)
        {
            if (apply_quantize)
            {
                var quantize_type = QuantizationType.Float16;
                ModelQuantizer.QuantizeWeights(quantize_type, ref runtime_model);
            }
            worker = WorkerFactory.CreateWorker(backend_type, runtime_model);
            backend = WorkerFactory.CreateBackend(backend_type);
            var input_width = runtime_model.inputs[0].shape[3].value;
            var input_height = runtime_model.inputs[0].shape[2].value;
            render_texture = RenderTexture.GetTemporary(input_width, input_height, 0, RenderTextureFormat.ARGBHalf);
            pre_process = new Material(Shader.Find("BaseModel/PreProcess"));
            pre_process.SetVector("_Mean", new Vector4(0.0f, 0.0f, 0.0f));
            pre_process.SetVector("_Std", new Vector4(1.0f, 1.0f, 1.0f));
            pre_process.SetFloat("_Max", 1.0f);
        }

        /// <summary>
        /// get input layers shape
        /// </summary>
        /// <returns>the input layers name with shape</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Dictionary<string, TensorShape> GetInputShapes()
        {
            var input_shapes = new Dictionary<string, TensorShape>(runtime_model.inputs.Count);
            runtime_model.inputs.ForEach(input =>
            {
                var shape = input.shape.ToTensorShape();
                input_shapes[input.name] = shape;
            });

            return input_shapes;
        }

        /// <summary>
        /// get output layers shape
        /// </summary>
        /// <returns>the output layers name with shape</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Dictionary<string, TensorShape> GetOutputShapes()
        {
            var input_shapes = GetInputShapes();
            var input_tensors = new Dictionary<string, Tensor>(runtime_model.inputs.Count);
            runtime_model.inputs.ForEach(input =>
            {
                var input_tensor = TensorFloat.AllocZeros(input.shape.ToTensorShape());
                input_tensors[input.name] = input_tensor;
            });

            worker.Execute(input_tensors);

            var output_shapes = new Dictionary<string, TensorShape>(runtime_model.outputs.Count);
            runtime_model.outputs.ForEach(output =>
                output_shapes[output.name] = worker.PeekOutput(output.name).shape
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

            backend?.Dispose();
            backend = WorkerFactory.CreateBackend(backend_type);
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
        /// set number of layers to process per frame
        /// </summary>
        /// <param name="layers_per_frame">number of layers per frame (-1 is process all layers per frame)</param>
        public void SetLayersPerFrame(int layers_per_frame)
        {
            this.layers_per_frame = layers_per_frame;
            if (this.layers_per_frame < 0)
            {
                this.layers_per_frame = runtime_model.layers.Count;
            }
        }

        /// <summary>
        /// run predict and get output tensors
        /// </summary>
        /// <param name="image">input image</param>
        /// <returns>output layers name with tensor</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Dictionary<string, Tensor> Predict(Texture2D image)
        {
            var input_texture = PreProcess(image);
            var input_shape = GetInputShapes().First().Value;
            var input_tensor = TextureConverter.ToTensor(input_texture, input_shape[3], input_shape[2], input_shape[1]);

            worker.Execute(input_tensor);

            var output_tensors = new Dictionary<string, Tensor>(runtime_model.outputs.Count);
            runtime_model.outputs.ForEach(output => {
                var output_tensor = worker.PeekOutput(output.name);
                output_tensors[output.name] = output_tensor;
            });

            input_tensor.Dispose();
            MonoBehaviour.Destroy(input_texture);

            return output_tensors;
        }

        /// <summary>
        /// run split predict over multiple frames and get output tensors
        /// </summary>
        /// <param name="image">input image</param>
        /// <param name="return_callback">return callback</param>
        /// <returns>callback function to returns output tensors</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public IEnumerator Predict(Texture2D image, Action<Dictionary<string, Tensor>> return_callback)
        {
            var input_texture = PreProcess(image);
            var input_shape = GetInputShapes().First().Value;
            var input_tensor = TextureConverter.ToTensor(input_texture, input_shape[3], input_shape[2], input_shape[1]);

            if (!is_predicting)
            {
                schedule = worker.ExecuteLayerByLayer(input_tensor);
                is_predicting = true;
            }

            var layers = 0;
            while (schedule.MoveNext())
            {
                if ((++layers % layers_per_frame) == 0)
                {
                    yield return null;
                }
            }

            var output_tensors = new Dictionary<string, Tensor>(runtime_model.outputs.Count);
            runtime_model.outputs.ForEach(output => {
                var output_tensor = worker.PeekOutput(output.name);
                output_tensors[output.name] = output_tensor;
            });

            input_tensor.Dispose();
            MonoBehaviour.Destroy(input_texture);

            is_predicting = false;

            return_callback(output_tensors);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Texture PreProcess(Texture2D image)
        {
            RenderTexture.active = render_texture;
            Graphics.Blit(image, render_texture, pre_process);

            var result = new Texture2D(render_texture.width, render_texture.height, TextureFormat.RGBAHalf, false);
            result.ReadPixels(new Rect(0, 0, render_texture.width, render_texture.height), 0, 0);
            result.Apply(false);

            return result;
        }
    }
}
