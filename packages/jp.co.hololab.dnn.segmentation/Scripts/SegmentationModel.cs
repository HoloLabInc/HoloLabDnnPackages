using System;
using System.Collections;
using System.Collections.Generic;
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
        /// create segmentation model for general segmentation models from sentis file
        /// </summary>
        /// <param name="file_path">model file path</param>
        /// <param name="backend_type">backend type for inference engine</param>
        public SegmentationModel(string file_path, BackendType backend_type = BackendType.GPUCompute)
            : base(file_path, backend_type)
        {
            Initialize();
        }

        /// <summary>
        /// create segmentation model for general segmentation models from stream
        /// </summary>
        /// <param name="stream">model stream</param>
        /// <param name="backend_type">backend type for inference engine</param>
        public SegmentationModel(System.IO.Stream stream, BackendType backend_type = BackendType.GPUCompute)
            : base(stream, backend_type)
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
            var output_name = runtime_model.outputs[0].name;
            var output_tensor = output_tensors[output_name] as Tensor<int>;

            output_tensor = output_tensor.ReadbackAndClone();

            var indices_texture = PostProcess(output_tensor, image.width, image.height);

            output_tensor.Dispose();
            output_tensors.AllDispose();

            return indices_texture;
        }

        /// <summary>
        /// segment area with split predict over multiple frames
        /// </summary>
        /// <param name="image">input image</param>
        /// <param name="return_callback">return callback</param>
        /// <returns>callback function to returns segment area texture with indices in color.r</returns>
        public IEnumerator Segment(Texture2D image, Action<Texture2D> return_callback)
        {
            var output_tensors = new Dictionary<string, Tensor>();
            yield return CoroutineHandler.StartStaticCoroutine(Predict(image, (outputs) => output_tensors = outputs));
            var output_name = runtime_model.outputs[0].name;
            var output_tensor = output_tensors[output_name] as Tensor<int>;

            output_tensor = output_tensor.ReadbackAndClone();

            var indices_texture = PostProcess(output_tensor, image.width, image.height);

            output_tensor.Dispose();
            output_tensors.AllDispose();

            return_callback(indices_texture);
        }

        /// <summary>
        /// get num classes
        /// </summary>
        /// <returns>num classes</returns>
        public int GetNumClasses()
        {
            var output_shapes = GetOutputShapes();
            var output_name = runtime_model.outputs[0].name;
            var num_classes = output_shapes[output_name][1];
            return num_classes;
        }

        private void Initialize()
        {
            SetInputMean(new Vector3(0.485f, 0.456f, 0.406f));
            SetInputStd(new Vector3(0.229f, 0.224f, 0.225f));
            SetEditedModel(AddPostProcess());
            SetSliceFrames(5); // TODO : automatic adjust number of layers per frame
        }

        private Model AddPostProcess()
        {
            try
            {
                var functional_graph = new FunctionalGraph();
                var inputs = functional_graph.AddInputs(base.runtime_model);
                var predict = Functional.Forward(base.runtime_model, inputs)[0];

                var indices_tensor = Functional.ArgMax(predict, 1).Squeeze();
                var fliped_tensor = indices_tensor.FlipUD();

                var edited_model = functional_graph.Compile(fliped_tensor);

                return edited_model;
            }
            catch (Exception e)
            {
                throw new Exception($"[error] can not add post process to model for some reason. ({e.Message})");
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Texture2D PostProcess(Tensor<int> tensor, int input_width, int input_height)
        {
            var indices_texture = ToTexture(tensor);
            var resized_texture = Resize(indices_texture, input_width, input_height);

            MonoBehaviour.Destroy(indices_texture);

            return resized_texture;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Texture2D ToTexture(Tensor<int> tensor)
        {
            var width = tensor.shape[1];
            var height = tensor.shape[0];
            var texture = new Texture2D(width, height, TextureFormat.R8, false);

            var indices = tensor.DownloadToNativeArray();
            var colors = new Color32[indices.Length];

            Parallel.For(0, width * height, i =>
            {
                colors[i].r = (byte)indices[i];
            });

            texture.SetPixels32(colors);
            texture.Apply(false);

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
            resized_texture.Apply(false);

            RenderTexture.active = null;
            RenderTexture.ReleaseTemporary(render_texture);

            return resized_texture;
        }
    }
}
