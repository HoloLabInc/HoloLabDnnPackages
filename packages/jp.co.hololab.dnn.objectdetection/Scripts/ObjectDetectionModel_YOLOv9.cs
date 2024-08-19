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
    /// object detection model class for yolo v9
    /// </summary>
    public class ObjectDetectionModel_YOLOv9 : BaseModel, IDisposable
    {
        private List<int> strides;
        private List<Vector2Int> anchors;
        private TensorFloat scalars;
        private TensorShape input_shape;

        /// <summary>
        /// create object detection model for yolox from sentis file
        /// </summary>
        /// <param name="file_path">model file path</param>
        /// <param name="backend_type">backend type for inference engine</param>
        /// <param name="apply_quantize">apply float16 quantize</param>
        public ObjectDetectionModel_YOLOv9(string file_path, BackendType backend_type = BackendType.GPUCompute, bool apply_quantize = true)
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
        public ObjectDetectionModel_YOLOv9(System.IO.Stream stream, BackendType backend_type = BackendType.GPUCompute, bool apply_quantize = true)
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
        public ObjectDetectionModel_YOLOv9(ModelAsset model_asset, BackendType backend_type = BackendType.GPUCompute, bool apply_quantize = true)
            : base(model_asset, backend_type, apply_quantize)
        {
            Initialize();
        }

        /// <summary>
        /// dispose object detection model
        /// </summary>
        public new void Dispose()
        {
            scalars?.Dispose();
            scalars = null;

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
            (var resize_ratio, var pad) = GetResizeRatio(image);

            var output_tensors = Predict(input_texture);

            var objects = PostProcess(output_tensors, resize_ratio, pad, score_threshold);
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
            (var resize_ratio, var pad) = GetResizeRatio(image);

            var output_tensors = new Dictionary<string, Tensor>();
            yield return CoroutineHandler.StartStaticCoroutine(Predict(input_texture, (outputs) => output_tensors = outputs));

            var objects = PostProcess(output_tensors, resize_ratio, pad, score_threshold);
            objects = NonMaximumSuppression.NMS(objects, score_threshold, iou_threshold);

            output_tensors.AllDispose();
            MonoBehaviour.Destroy(input_texture);

            return_callback(objects);
        }

        private void Initialize()
        {
            SetLayersPerFrame(runtime_model.layers.Count / 5);

            input_shape = GetInputShapes().First().Value;
            var input_width = input_shape[3];
            var input_height = input_shape[2];

            strides = CreateStrides();
            (anchors, scalars) = CreateAnchorsAndScalars(input_width, input_height);
        }

        private List<int> CreateStrides()
        {
            var strides = new List<int>();

            var input_shape = GetInputShapes().First().Value;
            var output_shapes = GetOutputShapes().Values.ToArray();

            for (var i = 2; i < output_shapes.Count(); i += 3)
            {
                strides.Add(input_shape[3] / output_shapes[i][3]);
            }

            return strides;
        }

        private (List<Vector2Int>, TensorFloat) CreateAnchorsAndScalars(int width, int height)
        {
            var anchors = new List<Vector2Int>();
            var scalars = new List<float>();

            foreach (var stride in strides)
            {
                var shift = stride / 2;
                var w = Enumerable.Range(0, width / stride).Select(i => (i * stride) + shift);
                var h = Enumerable.Range(0, height / stride).Select(i => (i * stride) + shift);
                var anchor = h.SelectMany(y => w.Select(x => new Vector2Int(x, y)));
                anchors.AddRange(anchor);

                var scalar = Enumerable.Repeat((float)stride, anchor.Count());
                scalars.AddRange(scalar);
            }

            return (anchors, new TensorFloat(new TensorShape(1, scalars.Count, 1), scalars.ToArray()));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private List<HoloLab.DNN.ObjectDetection.Object> PostProcess(Dictionary<string, Tensor> output_tensors, float resize_ratio, Vector2 pad, float score_threshold)
        {
            var predicts = output_tensors.Values.ToArray();
            var num_predicts = predicts.Count() / 3;
            var predicts_cls = new List<TensorFloat>(num_predicts);
            var predicts_box = new List<TensorFloat>(num_predicts);
            for (var i = 0; i < predicts.Count(); i+=3)
            {
                var predict_cls = predicts[i + 0] as TensorFloat;
                var predict_box = predicts[i + 2] as TensorFloat;
                predicts_cls.Add(Rearrange(predict_cls));
                predicts_box.Add(Rearrange(predict_box));
            }

            var classes_tensor = Concat(predicts_cls);
            var boxes_tensor = Mul(Concat(predicts_box), scalars);

            classes_tensor = classes_tensor.ReadbackAndClone();
            boxes_tensor = boxes_tensor.ReadbackAndClone();

            var classes = classes_tensor.ToReadOnlySpan();
            var boxes = boxes_tensor.ToReadOnlySpan();

            var num_objects = classes_tensor.shape[1];
            var num_classes = classes_tensor.shape[2];

            var objects = new List<HoloLab.DNN.ObjectDetection.Object>();
            for (var i = 0; i < num_objects; i++)
            {
                var confidences = classes.Slice(i * num_classes, num_classes).ToArray();
                confidences = confidences.Select(x => 1.0f / (1.0f + Mathf.Exp(-x))).ToArray();

                var confidence = confidences.Max();
                var class_id = Array.IndexOf(confidences, confidence);
                if (confidence < score_threshold)
                {
                    continue;
                }

                var anchor = anchors[i];
                var box = boxes.Slice(i * 4, 4);

                var x1y1 = anchor - new Vector2Int((int)box[0], (int)box[1]);
                var x2y2 = anchor + new Vector2Int((int)box[2], (int)box[3]);

                var x1 = (x1y1.x - pad.x) / resize_ratio;
                var y1 = (x1y1.y - pad.y) / resize_ratio;
                var x2 = (x2y2.x - pad.x) / resize_ratio;
                var y2 = (x2y2.y - pad.y) / resize_ratio;

                var rect = new Rect(x1, y1, x2 - x1, y2 - y1);

                objects.Add(new HoloLab.DNN.ObjectDetection.Object(rect, class_id, confidence));
            }

            classes_tensor.Dispose();
            boxes_tensor.Dispose();

            return objects;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private TensorFloat Rearrange(TensorFloat tensor)
        {
            var transpose_permutations = new int[4] { 0, 2, 3, 1 };
            var transpose_tensor = TensorFloat.AllocNoData(tensor.shape.Transpose(transpose_permutations));
            backend.Transpose(tensor, transpose_tensor, transpose_permutations);

            var reshape_shape = new int[3] { transpose_tensor.shape[0], transpose_tensor.shape[1] * transpose_tensor.shape[2], transpose_tensor.shape[3] };
            var reshape_tensor = TensorFloat.AllocNoData(new TensorShape(reshape_shape));
            backend.Reshape(transpose_tensor, reshape_tensor);

            transpose_tensor.Dispose();

            return reshape_tensor;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private TensorFloat Concat(List<TensorFloat> tensors, int axis = 1)
        {
            var base_shape = tensors.First().shape;
            base_shape[axis] = 0;
            var concat_shape = new TensorShape(base_shape);
            foreach (var tensor in tensors)
            {
                concat_shape = concat_shape.Concat(tensor.shape, axis);
            }
            var concat_tensor = TensorFloat.AllocNoData(concat_shape);

            var start = 0;
            foreach (var tensor in tensors)
            {
                backend.SliceSet(tensor, concat_tensor, axis, start, 1);
                start += tensor.shape[axis];
            }

            tensors.AllDispose();

            return concat_tensor;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private TensorFloat Mul(TensorFloat tensor, TensorFloat scalars)
        {
            var mul_tensor = TensorFloat.AllocNoData(tensor.shape);
            backend.Mul(tensor, scalars, mul_tensor);

            tensor.Dispose();

            return mul_tensor;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Texture2D ResizeSquare(Texture2D image)
        {
            var size = Math.Max(image.width, image.height);
            var result = new Texture2D(size, size, image.format, false);
            result.SetPixels(0, (size - image.height) / 2, image.width, image.height, image.GetPixels());
            result.Apply(false);

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private (float resize_ratio, Vector2 pad) GetResizeRatio(Texture2D image)
        {
            var input_width = input_shape[3];
            var input_height = input_shape[2];
            var resize_ratio = Math.Min((float)input_width / (float)image.width, (float)input_height / (float)image.height);

            var size = Math.Max((float)image.width, (float)image.height);
            var pad_x = ((size - image.width) * (input_width / size)) * 0.5f;
            var pad_y = ((size - image.height) * (input_height / size)) * 0.5f;

            return (resize_ratio, new Vector2(pad_x, pad_y));
        }
    }
}
