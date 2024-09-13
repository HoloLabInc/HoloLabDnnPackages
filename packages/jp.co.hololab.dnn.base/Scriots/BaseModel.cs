using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using UnityEngine;
using Unity.Sentis;

namespace HoloLab.DNN.Base
{
    /// <summary>
    /// dnn base model for image recognition tasks
    /// </summary>
    public class BaseModel : IDisposable
    {
        protected Model runtime_model = null;
        protected Worker worker = null;
        protected IEnumerator schedule = null;
        protected BackendType backend_type = BackendType.GPUCompute;
        protected bool is_quantized = false;
        private Material pre_process = null;
        private RenderTexture render_texture = null;
        private bool is_predicting = false;
        private int layers_per_frame = 5;

        /// <summary>
        /// create base model from sentis file
        /// </summary>
        /// <param name="file_path">model file path</param>
        /// <param name="backend_type">backend type for inference engine</param>
        public BaseModel(string file_path, BackendType backend_type = BackendType.GPUCompute)
        {
            this.runtime_model = ModelLoader.Load(file_path);
            this.backend_type = backend_type;
            InitializeWorker();
            InitializePreProcess();
        }

        /// <summary>
        /// create base model from stream
        /// </summary>
        /// <param name="stream">mdoel stream</param>
        /// <param name="backend_type">backend type for inference engine</param>
        public BaseModel(System.IO.Stream stream, BackendType backend_type = BackendType.GPUCompute)
        {
            this.runtime_model = ModelLoader.Load(stream);
            this.backend_type = backend_type;
            InitializeWorker();
            InitializePreProcess();
        }

        /// <summary>
        /// create base model from model asset
        /// </summary>
        /// <param name="model_asset">model asset</param>
        /// <param name="backend_type">backend type for inference engine</param>
        public BaseModel(ModelAsset model_asset, BackendType backend_type = BackendType.GPUCompute)
        {
            this.runtime_model = ModelLoader.Load(model_asset);
            this.backend_type = backend_type;
            InitializeWorker();
            InitializePreProcess();
        }

        /// <summary>
        /// dispose base model
        /// </summary>
        public void Dispose()
        {
            worker?.Dispose();
            worker = null;

            if (render_texture != null)
            {
                RenderTexture.ReleaseTemporary(render_texture);
                render_texture = null;
            }
        }

        private void InitializeWorker()
        {
            worker?.Dispose();
            worker = new Worker(runtime_model, backend_type);
            var input_width = runtime_model.inputs[0].shape.Get(3);
            var input_height = runtime_model.inputs[0].shape.Get(2);
            if (render_texture == null || render_texture.width != input_width || render_texture.height != input_height)
            {
                if (render_texture != null)
                {
                    RenderTexture.ReleaseTemporary(render_texture);
                }
                render_texture = RenderTexture.GetTemporary(input_width, input_height, 0, RenderTextureFormat.ARGBHalf);
            }
        }

        private void InitializePreProcess()
        {
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
            var input_tensors = new Dictionary<string, Tensor<float>>(runtime_model.inputs.Count);
            runtime_model.inputs.ForEach(input =>
            {
                var input_tensor = new Tensor<float>(input.shape.ToTensorShape());
                input_tensors[input.name] = input_tensor;
            });

            foreach (var input_tensor in input_tensors)
            {
                worker.SetInput(input_tensor.Key, input_tensor.Value);
            }
            worker.Schedule();

            var output_shapes = new Dictionary<string, TensorShape>(runtime_model.outputs.Count);
            runtime_model.outputs.ForEach(output =>
                output_shapes[output.name] = worker.PeekOutput(output.name).shape
            );

            return output_shapes;
        }

        /// <summary>
        /// set edited model by funtional api
        /// </summary>
        /// <param name="edited_model">edited model</param>
        public void SetEditedModel(Model edited_model)
        {
            this.runtime_model = edited_model;
            InitializeWorker();
        }

        /// <summary>
        /// set backend type for inference engine
        /// </summary>
        /// <param name="backend_type">backend type for inference engine</param>
        public void SetBackendType(BackendType backend_type)
        {
            this.backend_type = backend_type;
            InitializeWorker();
        }

        /// <summary>
        /// apply float16 quantize
        /// </summary>
        public void ApplyQuantize()
        {
            if (is_quantized)
            {
                return;
            }
            var quantize_type = QuantizationType.Float16;
            ModelQuantizer.QuantizeWeights(quantize_type, ref runtime_model);
            InitializeWorker();
            is_quantized = true;
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
        /// set number of layers per frame to slice inference
        /// </summary>
        /// <param name="num_layers">number of layers per frame (-1 is process all layers per frame)</param>
        [Obsolete("this method has been renamed. please use SetSliceLayers() or SetSliceFrames().", false)]
        public void SetLayersPerFrame(int num_layers)
        {
            SetSliceLayers(num_layers);
        }

        /// <summary>
        /// set number of layers per frame to slice inference
        /// </summary>
        /// <param name="num_layers">number of layers per frame (-1 is process all layers per frame)</param>
        public void SetSliceLayers(int num_layers)
        {
            this.layers_per_frame = num_layers;
            if (this.layers_per_frame <= 0)
            {
                this.layers_per_frame = runtime_model.layers.Count;
            }
        }

        /// <summary>
        /// set number of frames to slice inference
        /// </summary>
        /// <param name="num_frames">number of frames per inference (-1 is process all layers per frame)</param>
        public void SetSliceFrames(int num_frames)
        {
            try
            {
                this.layers_per_frame = (int)Math.Ceiling((float)runtime_model.layers.Count / (float)num_frames);
            }
            finally
            {
                if (this.layers_per_frame <= 0)
                {
                    this.layers_per_frame = runtime_model.layers.Count;
                }
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

            worker.Schedule(input_tensor);

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
                schedule = worker.ScheduleIterable(input_tensor);
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
