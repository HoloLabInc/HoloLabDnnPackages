using System;
using System.Threading.Tasks;
using System.Runtime.CompilerServices;
using UnityEngine;
using Unity.Sentis;
using HoloLab.DNN.Base;

namespace HoloLab.DNN.Segmentation
{
    /// <summary>
    /// segmentation model class for general segmentation models
    /// (this class supports models with output shape is 1 x num_classes x height x width)
    /// </summary>
    public class SegmentationModel : BaseModel, IDisposable
    {
        /// <summary>
        /// create segmentation model for general segmentation models from onnx file
        /// </summary>
        /// <param name="file_path">model file path</param>
        /// <param name="backend_type">backend type for inference engine</param>
        public SegmentationModel(string file_path, BackendType backend_type = BackendType.GPUCompute)
            : base(file_path, backend_type)
        {
            Initialize();
        }

        /// <summary>
        /// create segmentation model for general segmentation models from model asset
        /// </summary>
        /// <param name="model_asset">model asset</param>
        /// <param name="backend_type">backend type for inference engine</param>
        public SegmentationModel(ModelAsset model_asset, BackendType backend_type = BackendType.GPUCompute)
            : base(model_asset, backend_type)
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
        /// segment area
        /// </summary>
        /// <param name="image">input image</param>
        /// <returns>segment area texture with indices in color.r</returns>
        public Texture2D Segment(Texture2D image)
        {
            var output_tensors = Predict(image);
            var output_name = runtime_model.outputs[0];
            var output_tensor = output_tensors[output_name] as TensorFloat;
            var indices = ops.ArgMax(output_tensor, 1, false);

            output_tensor.MakeReadable();
            indices.MakeReadable();

            var indices_texture = ToTexture(indices);
            var resized_texture = Resize(indices_texture, image.width, image.height);

            output_tensors.AllDispose();
            indices.Dispose();
            MonoBehaviour.Destroy(indices_texture);

            return resized_texture;
        }

        /// <summary>
        /// get num classes
        /// </summary>
        /// <returns>num classes</returns>
        public int GetNumClasses()
        {
            var output_shapes = GetOutputShapes();
            var output_name = runtime_model.outputs[0];
            var num_classes = output_shapes[output_name][1];
            return num_classes;
        }

        private void Initialize()
        {
            SetInputMean(new Vector3(0.485f, 0.456f, 0.406f));
            SetInputStd(new Vector3(0.229f, 0.224f, 0.225f));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Texture2D ToTexture(TensorInt tensor)
        {
            var width = tensor.shape[2];
            var height = tensor.shape[1];
            var texture = new Texture2D(width, height, TextureFormat.R8, false);

            var indices = tensor.ToReadOnlyArray();
            var colors = new Color32[indices.Length];

            Parallel.For(0, height, y =>
            {
                var inv_y = height - 1 - y;
                for (var x = 0; x < width; x++)
                {
                    var index = (byte)indices[inv_y * width + x];
                    colors[y * width + x] = new Color32(index, 0, 0, 255);
                }
            });

            texture.SetPixels32(colors);
            texture.Apply();

            return texture;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Texture2D Resize(Texture2D texture, int width, int height, FilterMode filter_mode = FilterMode.Point)
        {
            var render_texture = RenderTexture.GetTemporary(width, height, 0, RenderTextureFormat.R8);
            render_texture.filterMode = filter_mode;

            texture.filterMode = filter_mode;

            RenderTexture.active = render_texture;
            Graphics.Blit(texture, render_texture);

            var resized_texture = new Texture2D(render_texture.width, render_texture.height, TextureFormat.R8, false);
            resized_texture.ReadPixels(new Rect(0, 0, render_texture.width, render_texture.height), 0, 0);
            resized_texture.Apply();

            RenderTexture.active = null;
            RenderTexture.ReleaseTemporary(render_texture);

            return resized_texture;
        }
    }
}
