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
        private Tensor<float> scalars;
        private TensorShape input_shape;
        private TensorShape[] output_shapes;

        /// <summary>
        /// create object detection model for yolo v9 from sentis file
        /// </summary>
        /// <param name="file_path">model file path</param>
        /// <param name="backend_type">backend type for inference engine</param>
        public ObjectDetectionModel_YOLOv9(string file_path, BackendType backend_type = BackendType.GPUCompute)
            : base(file_path, backend_type)
        {
            Initialize();
        }

        /// <summary>
        /// create object detection model for yolo v9 from stream
        /// </summary>
        /// <param name="stream">model stream</param>
        /// <param name="backend_type">backend type for inference engine</param>
        public ObjectDetectionModel_YOLOv9(System.IO.Stream stream, BackendType backend_type = BackendType.GPUCompute)
            : base(stream, backend_type)
        {
            Initialize();
        }

        /// <summary>
        /// create object detection model for yolo v9 from model asset
        /// </summary>
        /// <param name="model_asset">model asset</param>
        /// <param name="backend_type">backend type for inference engine</param>
        public ObjectDetectionModel_YOLOv9(ModelAsset model_asset, BackendType backend_type = BackendType.GPUCompute)
            : base(model_asset, backend_type)
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
        public List<HoloLab.DNN.ObjectDetection.BoundingBox> Detect(Texture2D image, float score_threshold = 0.5f, float iou_threshold = 0.4f)
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
        public IEnumerator Detect(Texture2D image, float score_threshold, float iou_threshold, Action<List<HoloLab.DNN.ObjectDetection.BoundingBox>> return_callback)
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
            input_shape = GetInputShapes().First().Value;
            output_shapes = GetOutputShapes().Values.ToArray();
            var input_width = input_shape[3];
            var input_height = input_shape[2];

            strides = CreateStrides();
            (anchors, scalars) = CreateAnchorsAndScalars(input_width, input_height);

            SetEditedModel(AddPostProcess());
            SetSliceFrames(5); // TODO : automatic adjust number of layers per frame
        }

        private Model AddPostProcess()
        {
            try
            {
                var functional_graph = new FunctionalGraph();
                var inputs = functional_graph.AddInputs(base.runtime_model);
                var predicts = Functional.Forward(base.runtime_model, inputs);

                var num_predicts = predicts.Count() / 3;
                var predicts_cls = new List<FunctionalTensor>(num_predicts);
                var predicts_box = new List<FunctionalTensor>(num_predicts);
                for (var i = 0; i < predicts.Count(); i += 3)
                {
                    var predict_cls = predicts[i + 0];
                    var predict_cls_shape = output_shapes[i + 0];
                    var transpose_cls = predict_cls.Transpose(1, 2).Transpose(2, 3);
                    var reshape_cls = transpose_cls.Reshape(new int[3] { predict_cls_shape[0], predict_cls_shape[2] * predict_cls_shape[3], predict_cls_shape[1] });
                    predicts_cls.Add(reshape_cls);

                    var predict_box = predicts[i + 2];
                    var predict_box_shape = output_shapes[i + 2];
                    var transpose_box = predict_box.Transpose(1, 2).Transpose(2, 3);
                    var reshape_box = transpose_box.Reshape(new int[3] { predict_box_shape[0], predict_box_shape[2] * predict_box_shape[3], predict_box_shape[1] });
                    predicts_box.Add(reshape_box);
                }

                var classes_tensor = Functional.Concat(predicts_cls.ToArray(), 1);

                var scalers_tensor = Functional.Constant(scalars);
                var boxes_tensor = Functional.Mul(Functional.Concat(predicts_box.ToArray(), 1), scalers_tensor);

                var edited_model = functional_graph.Compile(classes_tensor, boxes_tensor);

                return edited_model;
            }
            catch (Exception e)
            {
                throw new Exception($"[error] can not add post process to model for some reason. ({e.Message})");
            }
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

        private (List<Vector2Int>, Tensor<float>) CreateAnchorsAndScalars(int width, int height)
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

            return (anchors, new Tensor<float>(new TensorShape(1, scalars.Count, 1), scalars.ToArray()));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private List<HoloLab.DNN.ObjectDetection.BoundingBox> PostProcess(Dictionary<string, Tensor> output_tensors, float resize_ratio, Vector2 pad, float score_threshold)
        {
            var predict_tensors = output_tensors.Values.ToArray();

            var classes_tensor = predict_tensors[0].ReadbackAndClone() as Tensor<float>;
            var boxes_tensor = predict_tensors[1].ReadbackAndClone() as Tensor<float>;

            var classes = classes_tensor.AsReadOnlySpan();
            var boxes = boxes_tensor.AsReadOnlySpan();

            var num_objects = classes_tensor.shape[1];
            var num_classes = classes_tensor.shape[2];

            var objects = new List<HoloLab.DNN.ObjectDetection.BoundingBox>();
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

                objects.Add(new HoloLab.DNN.ObjectDetection.BoundingBox(rect, class_id, confidence));
            }

            classes_tensor.Dispose();
            boxes_tensor.Dispose();

            return objects;
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
