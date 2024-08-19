using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using UnityEngine;
using Unity.Sentis;
using HoloLab.DNN.Base;

namespace HoloLab.DNN.ObjectDetection
{
    /// <summary>
    /// object detection model class for yolox
    /// </summary>
    public class ObjectDetectionModel_YOLOX : BaseModel, IDisposable
    {
        private List<int> strides = new List<int> { 8, 16, 32, 64 };
        private List<Vector2Int> grids;
        private List<int> expanded_strides;
        private TensorShape input_shape;

        /// <summary>
        /// create object detection model for yolox from sentis file
        /// </summary>
        /// <param name="file_path">model file path</param>
        /// <param name="backend_type">backend type for inference engine</param>
        /// <param name="apply_quantize">apply float16 quantize</param>
        public ObjectDetectionModel_YOLOX(string file_path, BackendType backend_type = BackendType.GPUCompute, bool apply_quantize = true)
            : base(file_path, backend_type, apply_quantize)
        {
            Initialize();
        }

        /// <summary>
        /// create object detection model for yolox from stream
        /// </summary>
        /// <param name="stream">model stream</param>
        /// <param name="backend_type">backend type for inference engine</param>
        /// <param name="apply_quantize">apply float16 quantize</param>
        public ObjectDetectionModel_YOLOX(System.IO.Stream stream, BackendType backend_type = BackendType.GPUCompute, bool apply_quantize = true)
            : base(stream, backend_type, apply_quantize)
        {
            Initialize();
        }

        /// <summary>
        /// create object detection model for yolox from model asset
        /// </summary>
        /// <param name="model_asset">model asset</param>
        /// <param name="backend_type">backend type for inference engine</param>
        /// <param name="apply_quantize">apply float16 quantize</param>
        public ObjectDetectionModel_YOLOX(ModelAsset model_asset, BackendType backend_type = BackendType.GPUCompute, bool apply_quantize = true)
            : base(model_asset, backend_type, apply_quantize)
        {
            Initialize();
        }

        /// <summary>
        /// dispose object detection model
        /// </summary>
        public new void Dispose()
        {
            base.Dispose();
        }

        /// <summary>
        /// detect objects
        /// </summary>
        /// <param name="image">input image</param>
        /// <param name="score_threshold">confidence score threshold</param>
        /// <param name="iou_threshold">iou threshold</param>
        /// <returns>detected object list</returns>
        public List<HoloLab.DNN.ObjectDetection.Object> Detect(Texture2D image, float score_threshold = 0.5f, float iou_threshold = 0.4f)
        {
            var input_texture = ResizeSquare(image);
            var resize_ratio = GetResizeRatio(image);

            var output_tensors = Predict(input_texture);
            var output_name = runtime_model.outputs[0].name;
            var output_tensor = output_tensors[output_name] as TensorFloat;

            var objects = PostProcess(output_tensor, resize_ratio, score_threshold);
            objects = NonMaximumSuppression.NMS(objects, score_threshold, iou_threshold);

            output_tensors.AllDispose();
            MonoBehaviour.Destroy(input_texture);

            return objects;
        }

        /// <summary>
        /// detect objects with split predict over multiple frames
        /// </summary>
        /// <param name="image">input image</param>
        /// <param name="score_threshold">confidence score threshold</param>
        /// <param name="iou_threshold">iou threshold</param>
        /// <param name="return_callback">return callback</param>
        /// <returns>callback function to returns detected object list</returns>
        public IEnumerator Detect(Texture2D image, float score_threshold, float iou_threshold, Action<List<HoloLab.DNN.ObjectDetection.Object>> return_callback)
        {
            var input_texture = ResizeSquare(image);
            var resize_ratio = GetResizeRatio(image);

            var output_tensors = new Dictionary<string, Tensor>();
            yield return CoroutineHandler.StartStaticCoroutine(Predict(input_texture, (outputs) => output_tensors = outputs));
            var output_name = runtime_model.outputs[0].name;
            var output_tensor = output_tensors[output_name] as TensorFloat;

            var objects = PostProcess(output_tensor, resize_ratio, score_threshold);
            objects = NonMaximumSuppression.NMS(objects, score_threshold, iou_threshold);

            output_tensors.AllDispose();
            MonoBehaviour.Destroy(input_texture);

            return_callback(objects);
        }

        private void Initialize()
        {
            SetInputMax(255.0f);
            SetLayersPerFrame(runtime_model.layers.Count / 5); // TODO : automatic adjust number of layers per frame

            input_shape = GetInputShapes().First().Value;
            var input_width = input_shape[3];
            var input_height = input_shape[2];

            var wsizes = strides.Select(stride => input_width / stride).ToList();
            var hsizes = strides.Select(stride => input_height / stride).ToList();
            (grids, expanded_strides) = CreateGridsAndExpandedStrides(wsizes, hsizes);
        }

        private (List<Vector2Int>, List<int>) CreateGridsAndExpandedStrides(List<int> wsizes, List<int> hsizes)
        {
            var grids = new List<Vector2Int>();
            var expanded_strides = new List<int>();

            for (int i = 0; i < strides.Count; i++)
            {
                (var wsize, var hsize, var stride) = (wsizes[i], hsizes[i], strides[i]);
                for (int y = 0; y < hsize; y++)
                {
                    for (int x = 0; x < wsize; x++)
                    {
                        grids.Add(new Vector2Int(x, y));
                        expanded_strides.Add(stride);
                    }
                }
            }

            return (grids, expanded_strides);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Texture2D ResizeSquare(Texture2D image)
        {
            var size = Math.Max(image.width, image.height);
            var result = new Texture2D(size, size, image.format, false);
            result.SetPixels(0, size - image.height, image.width, image.height, image.GetPixels());
            result.Apply(false);

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float GetResizeRatio(Texture2D image)
        {
            var input_width = input_shape[3];
            var input_height = input_shape[2];
            return Math.Min((float)input_width / (float)image.width, (float)input_height / (float)image.height);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private List<HoloLab.DNN.ObjectDetection.Object> PostProcess(TensorFloat output_tensor, float resize_ratio, float score_threshold = 0.0f)
        {
            output_tensor = output_tensor.ReadbackAndClone();
            var output_span = output_tensor.ToReadOnlySpan();

            var objects = new List<HoloLab.DNN.ObjectDetection.Object>();
            var tensor_width = output_tensor.shape[2];
            var tensor_channels = output_tensor.shape[1];
            for (int c = 0; c < tensor_channels; c++)
            {
                var span = output_span.Slice(c * tensor_width, tensor_width);

                var objectness = span[4];
                var confidences = span.Slice(5);
                var confidence = confidences.ToArray().Max();
                var score = confidence * objectness;
                if (score < score_threshold) { continue; }
                var class_id = confidences.IndexOf(confidence);

                var grid = grids[c];
                var expanded_stride = expanded_strides[c];

                var center_x = ((span[0] + grid.x) * expanded_stride) / resize_ratio;
                var center_y = ((span[1] + grid.y) * expanded_stride) / resize_ratio;
                var width = (float)(Math.Exp(span[2]) * expanded_stride) / resize_ratio;
                var height = (float)(Math.Exp(span[3]) * expanded_stride) / resize_ratio;
                var rect = new Rect(center_x - (width * 0.5f), center_y - (height * 0.5f), width, height);

                objects.Add(new HoloLab.DNN.ObjectDetection.Object(rect, class_id, score));
            }

            output_tensor.Dispose();

            return objects;
        }
    }
}
