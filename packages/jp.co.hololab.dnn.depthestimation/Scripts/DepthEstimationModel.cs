using System;
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
        /// create depth estimation model for midas from onnx file
        /// </summary>
        /// <param name="file_path">model file path</param>
        /// <param name="backend_type">backend type for inference engine</param>
        public DepthEstimationModel(string file_path, BackendType backend_type = BackendType.GPUCompute)
            : base(file_path, backend_type)
        {
        }

        /// <summary>
        /// create depth estimation model for midas from model asset
        /// </summary>
        /// <param name="model_asset">model asset</param>
        /// <param name="backend_type">backend type for inference engine</param>
        public DepthEstimationModel(ModelAsset model_asset, BackendType backend_type = BackendType.GPUCompute)
            : base(model_asset, backend_type)
        {
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
            var output_name = runtime_model.outputs.Count == 1 ? runtime_model.outputs[0] : "output"; // TODO : fixed output layer name, because only dpt_levit_224_224x224 model have 2 outputs. (maybe bug)
            var output_tensor = output_tensors[output_name] as TensorFloat;

            output_tensor.MakeReadable();

            var depth_texture = PostProcess(output_tensor, image.width, image.height);

            output_tensors.AllDispose();

            return depth_texture;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Texture2D PostProcess(TensorFloat tensor, int input_width, int input_height)
        {
            var normalized_tensor = Normalize(tensor);
            var render_texture = ToRenderTexture(normalized_tensor);
            var depth_texture = Resize(render_texture, input_width, input_height);

            normalized_tensor.Dispose();

            return depth_texture;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private TensorFloat Normalize(TensorFloat tensor)
        {
            var min_tensor = ops.ReduceMin(tensor, null, false);
            var max_tensor = ops.ReduceMax(tensor, null, false);
            var normalized_tensor = ops.Div(ops.Sub(tensor, min_tensor), ops.Sub(max_tensor, min_tensor));

            min_tensor.Dispose();
            max_tensor.Dispose();

            return normalized_tensor;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private RenderTexture ToRenderTexture(TensorFloat tensor)
        {
            if (tensor.shape.rank != 4)
            {
                var tensor_shape = tensor.shape.Unsqueeze(0);
                tensor = tensor.ShallowReshape(tensor_shape) as TensorFloat;
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
            resized_texture.Apply();

            RenderTexture.active = null;
            RenderTexture.ReleaseTemporary(render_texture);

            return resized_texture;
        }
    }
}
