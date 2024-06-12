using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using UnityEngine;
using Unity.Sentis;
using HoloLab.DNN.Base;

namespace HoloLab.DNN.DepthEstimation
{
    /// <summary>
    /// depth estimation model class for general relative depth estimation models
    /// (this class supports models with output shape is 1 x 1 x height x width)
    /// </summary>
    public class DepthEstimationModel : BaseModel, IDisposable
    {
        /// <summary>
        /// create depth estimation model for midas from sentis file
        /// </summary>
        /// <param name="file_path">model file path</param>
        /// <param name="backend_type">backend type for inference engine</param>
        /// <param name="apply_quantize">apply float16 quantize</param>
        public DepthEstimationModel(string file_path, BackendType backend_type = BackendType.GPUCompute, bool apply_quantize = true)
            : base(file_path, backend_type, apply_quantize)
        {
            Initialize();
        }

        /// <summary>
        /// create depth estimation model for midas from stream
        /// </summary>
        /// <param name="stream">model stream</param>
        /// <param name="backend_type">backend type for inference engine</param>
        /// <param name="apply_quantize">apply float16 quantize</param>
        public DepthEstimationModel(System.IO.Stream stream, BackendType backend_type = BackendType.GPUCompute, bool apply_quantize = true)
            : base(stream, backend_type, apply_quantize)
        {
            Initialize();
        }

        /// <summary>
        /// create depth estimation model for midas from model asset
        /// </summary>
        /// <param name="model_asset">model asset</param>
        /// <param name="backend_type">backend type for inference engine</param>
        /// <param name="apply_quantize">apply float16 quantize</param>
        public DepthEstimationModel(ModelAsset model_asset, BackendType backend_type = BackendType.GPUCompute, bool apply_quantize = true)
            : base(model_asset, backend_type, apply_quantize)
        {
            Initialize();
        }

        /// <summary>
        /// dispose depth estimation model
        /// </summary>
        public new void Dispose()
        {
            base.Dispose();
        }

        /// <summary>
        /// estimate depth
        /// </summary>
        /// <param name="image">input image</param>
        /// <returns>estimated depth image that min-max normalized</returns>
        public Texture2D Estimate(Texture2D image)
        {
            var output_tensors = Predict(image);
            var output_name = runtime_model.outputs.Count == 1 ? runtime_model.outputs[0].name : "output"; // TODO : fixed output layer name, because only dpt_levit_224_224x224 model have 2 outputs. (maybe bug)
            var output_tensor = output_tensors[output_name] as TensorFloat;

            output_tensor.CompleteOperationsAndDownload();

            var depth_texture = PostProcess(output_tensor, image.width, image.height);

            output_tensors.AllDispose();

            return depth_texture;
        }

        /// <summary>
        /// estimate depth with split predict over multiple frames
        /// </summary>
        /// <param name="image">input image</param>
        /// <param name="return_callback">return callback</param>
        /// <returns>callback function to returns estimated depth image that min-max normalized</returns>
        public IEnumerator Estimate(Texture2D image, Action<Texture2D> return_callback)
        {
            var output_tensors = new Dictionary<string, Tensor>();
            yield return CoroutineHandler.StartStaticCoroutine(Predict(image, (outputs) => output_tensors = outputs));
            var output_name = runtime_model.outputs.Count == 1 ? runtime_model.outputs[0].name : "output"; // TODO : fixed output layer name, because only dpt_levit_224_224x224 model have 2 outputs. (maybe bug)
            var output_tensor = output_tensors[output_name] as TensorFloat;

            output_tensor.CompleteOperationsAndDownload();

            var depth_texture = PostProcess(output_tensor, image.width, image.height);

            output_tensors.AllDispose();

            return_callback(depth_texture);
        }

        private void Initialize()
        {
            SetLayersPerFrame(runtime_model.layers.Count / 5); // TODO : automatic adjust number of layers per frame
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Texture2D PostProcess(TensorFloat tensor, int input_width, int input_height)
        {
            var normalized_tensor = Normalize(tensor);
            var render_texture = ToRenderTexture(normalized_tensor);
            var depth_texture = Resize(render_texture, input_width, input_height);

            normalized_tensor?.Dispose();

            return depth_texture;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private TensorFloat Normalize(TensorFloat tensor)
        {
            var min_tensor = TensorFloat.AllocNoData(new TensorShape(1));
            backend.ReduceMin(tensor, min_tensor, null, false);
            var max_tensor = TensorFloat.AllocNoData(new TensorShape(1));
            backend.ReduceMax(tensor, max_tensor, null, false);

            var numerator_tensor = TensorFloat.AllocNoData(tensor.shape);
            var denominator_tensor = TensorFloat.AllocNoData(tensor.shape);
            var normalized_tensor = TensorFloat.AllocNoData(tensor.shape);
            backend.Sub(tensor, min_tensor, numerator_tensor);
            backend.Sub(max_tensor, min_tensor, denominator_tensor);
            backend.Div(numerator_tensor, denominator_tensor, normalized_tensor);

            min_tensor?.Dispose();
            max_tensor?.Dispose();
            numerator_tensor?.Dispose();
            denominator_tensor?.Dispose();

            return normalized_tensor;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private RenderTexture ToRenderTexture(TensorFloat tensor)
        {
            if (tensor.shape.rank != 4)
            {
                var tensor_shape = tensor.shape.Unsqueeze(0);
                tensor.Reshape(tensor_shape);
            }

            var width = tensor.shape[2];
            var height = tensor.shape[3];
            var channels = tensor.shape[1];
            return TextureConverter.ToTexture(tensor, width, height, channels);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Texture2D Resize(RenderTexture texture, int width, int height)
        {
            var render_texture = RenderTexture.GetTemporary(width, height, 0, RenderTextureFormat.ARGB32);

            RenderTexture.active = render_texture;
            Graphics.Blit(texture, render_texture);

            var resized_texture = new Texture2D(render_texture.width, render_texture.height, TextureFormat.RGBA32, false);
            resized_texture.ReadPixels(new Rect(0, 0, resized_texture.width, resized_texture.height), 0, 0);
            resized_texture.Apply(false);

            RenderTexture.active = null;
            RenderTexture.ReleaseTemporary(render_texture);

            return resized_texture;
        }
    }
}
