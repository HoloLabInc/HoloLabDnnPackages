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
    /// object detection model class for detr
    /// </summary>
    public class ObjectDetectionModel_DETR : BaseModel, IDisposable
    {
        private new RenderTexture render_texture = null;
        private TensorShape input_shape;
        private TensorShape[] output_shapes;

        /// <summary>
        /// create object detection model for detr from sentis file
        /// </summary>
        /// <param name="file_path">model file path</param>
        /// <param name="backend_type">backend type for inference engine</param>
        public ObjectDetectionModel_DETR(string file_path, BackendType backend_type = BackendType.GPUCompute)
            : base(file_path, backend_type)
        {
            Initialize();
        }

        /// <summary>
        /// create object detection model for detr from stream
        /// </summary>
        /// <param name="stream">model stream</param>
        /// <param name="backend_type">backend type for inference engine</param>
        public ObjectDetectionModel_DETR(System.IO.Stream stream, BackendType backend_type = BackendType.GPUCompute)
            : base(stream, backend_type)
        {
            Initialize();
        }

        /// <summary>
        /// create object detection model for detr from model asset
        /// </summary>
        /// <param name="model_asset">model asset</param>
        /// <param name="backend_type">backend type for inference engine</param>
        public ObjectDetectionModel_DETR(ModelAsset model_asset, BackendType backend_type = BackendType.GPUCompute)
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

            if (render_texture != null)
            {
                RenderTexture.ReleaseTemporary(render_texture);
                render_texture = null;
            }
        }

        /// <summary>
        /// detect objects
        /// </summary>
        /// <param name="image">input image</param>
        /// <param name="score_threshold">confidence score threshold</param>
        /// <param name="iou_threshold">iou threshold (not used in this class)</param>
        /// <returns>detected object list</returns>
        /// <remarks>iou_threshold arguments for compatibility with other object detection model apis. iou_threshold arguments ignored in this class.</remarks>
        public List<HoloLab.DNN.ObjectDetection.Object> Detect(Texture2D image, float score_threshold = 0.5f, float iou_threshold = 0.0f)
        {
            var output_tensors = Predict(image);

            var objects = PostProcess(output_tensors, score_threshold);

            output_tensors.AllDispose();
            return objects;
        }

        /// <summary>
        /// detect objects with split predict over multiple frames
        /// </summary>
        /// <param name="image">input image</param>
        /// <param name="score_threshold">confidence score threshold</param>
        /// <param name="iou_threshold">iou threshold (not used in this class)</param>
        /// <param name="return_callback">return callback</param>
        /// <returns>detected object list</returns>
        /// <remarks>iou_threshold arguments for compatibility with other object detection model apis. iou_threshold arguments ignored in this class.</remarks>
        public IEnumerator Detect(Texture2D image, float score_threshold, float iou_threshold, Action<List<HoloLab.DNN.ObjectDetection.Object>> return_callback)
        {
            var output_tensors = new Dictionary<string, Tensor>();
            yield return CoroutineHandler.StartStaticCoroutine(Predict(image, (outputs) => output_tensors = outputs));

            var objects = PostProcess(output_tensors, score_threshold);

            output_tensors.AllDispose();
            return_callback(objects);
        }

        private void Initialize()
        {
            input_shape = GetInputShapes().First().Value;
            output_shapes = GetOutputShapes().Values.ToArray();

            if (render_texture == null || render_texture.width != input_shape[3] || render_texture.height != input_shape[2])
            {
                if (render_texture != null)
                {
                    RenderTexture.ReleaseTemporary(render_texture);
                }
                render_texture = RenderTexture.GetTemporary(input_shape[3], input_shape[2], 0, RenderTextureFormat.ARGBHalf);
            }

            SetSliceFrames(5); // TODO : automatic adjust number of layers per frame
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public new Dictionary<string, Tensor> Predict(Texture2D image)
        {
            var input_texture = Resize(image);
            var images = TextureConverter.ToTensor(input_texture, input_shape[3], input_shape[2], input_shape[1]);
            worker.SetInput("images", images);

            var orig_target_sizes = new Tensor<int>(new TensorShape(1, 2), new int[] { image.width, image.height });
            worker.SetInput("orig_target_sizes", orig_target_sizes);

            worker.Schedule();

            var output_tensors = new Dictionary<string, Tensor>(runtime_model.outputs.Count);
            runtime_model.outputs.ForEach(output =>
            {
                var output_tensor = worker.PeekOutput(output.name);
                output_tensors[output.name] = output_tensor;
            });

            images.Dispose();
            orig_target_sizes.Dispose();
            MonoBehaviour.Destroy(input_texture);

            return output_tensors;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public new IEnumerator Predict(Texture2D image, Action<Dictionary<string, Tensor>> return_callback)
        {
            var input_texture = Resize(image);
            var images = TextureConverter.ToTensor(input_texture, input_shape[3], input_shape[2], input_shape[1]);
            worker.SetInput("images", images);

            var orig_target_sizes = new Tensor<int>(new TensorShape(1, 2), new int[] { image.width, image.height });
            worker.SetInput("orig_target_sizes", orig_target_sizes);

            if (!is_predicting)
            {
                schedule = worker.ScheduleIterable();
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
            runtime_model.outputs.ForEach(output =>
            {
                var output_tensor = worker.PeekOutput(output.name);
                output_tensors[output.name] = output_tensor;
            });

            images.Dispose();
            orig_target_sizes.Dispose();
            MonoBehaviour.Destroy(input_texture);

            is_predicting = false;

            return_callback(output_tensors);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Texture2D Resize(Texture2D image)
        {
            RenderTexture.active = render_texture;
            Graphics.Blit(image, render_texture);

            var result = new Texture2D(render_texture.width, render_texture.height, TextureFormat.RGBAHalf, false);
            result.ReadPixels(new Rect(0, 0, render_texture.width, render_texture.height), 0, 0);
            result.Apply(false);

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private List<HoloLab.DNN.ObjectDetection.Object> PostProcess(Dictionary<string, Tensor> output_tensors, float score_threshold)
        {
            var labels_tensor = output_tensors["labels"].ReadbackAndClone() as Tensor<int>;
            var boxes_tensor = output_tensors["boxes"].ReadbackAndClone() as Tensor<float>;
            var scores_tensor = output_tensors["scores"].ReadbackAndClone() as Tensor<float>;

            var labels = labels_tensor.AsReadOnlyNativeArray();
            var boxes = boxes_tensor.AsReadOnlyNativeArray();
            var scores = scores_tensor.AsReadOnlyNativeArray();

            var objects = new List<HoloLab.DNN.ObjectDetection.Object>();
            var num_detects = labels.Length;
            for (var i = 0; i < num_detects; i++)
            {
                var confidence = scores[i];
                if (confidence < score_threshold)
                {
                    continue;
                }

                var class_id = labels[i];

                var index = i * 4;
                var x1 = boxes[index + 0];
                var y1 = boxes[index + 1];
                var x2 = boxes[index + 2];
                var y2 = boxes[index + 3];
                var rect = new Rect(x1, y1, x2 - x1, y2 - y1);

                objects.Add(new HoloLab.DNN.ObjectDetection.Object(rect, class_id, confidence));
            }

            labels_tensor.Dispose();
            boxes_tensor.Dispose();
            scores_tensor.Dispose();

            return objects;
        }
    }
}
