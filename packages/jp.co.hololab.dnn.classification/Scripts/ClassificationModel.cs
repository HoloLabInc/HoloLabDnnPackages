using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis;
using HoloLab.DNN.Base;

namespace HoloLab.DNN.Classification
{
    /// <summary>
    /// classification model class for general classification models
    /// (this class supports models with output shape is 1 x num_classes)
    /// </summary>
    public class ClassificationModel : BaseModel, IDisposable
    {
        private bool apply_softmax = true;

        /// <summary>
        /// create classification model from sentis file
        /// </summary>
        /// <param name="file_path">model file path</param>
        /// <param name="backend_type">backend type for inference engine</param>
        /// <param name="apply_quantize">apply float16 quantize</param>
        public ClassificationModel(string file_path, BackendType backend_type = BackendType.GPUCompute, bool apply_quantize = true)
            : base(file_path, backend_type, apply_quantize)
        {
            Initialize();
        }

        /// <summary>
        /// create classification model from stream
        /// </summary>
        /// <param name="stream">model stream</param>
        /// <param name="backend_type">backend type for inference engine</param>
        /// <param name="apply_quantize">apply float16 quantize</param>
        public ClassificationModel(System.IO.Stream stream, BackendType backend_type = BackendType.GPUCompute, bool apply_quantize = true)
            : base(stream, backend_type, apply_quantize)
        {
            Initialize();
        }

        /// <summary>
        /// create classification model from model asset
        /// </summary>
        /// <param name="model_asset">model asset</param>
        /// <param name="backend_type">backend type for inference engine</param>
        /// <param name="apply_quantize">apply float16 quantize</param>
        public ClassificationModel(ModelAsset model_asset, BackendType backend_type = BackendType.GPUCompute, bool apply_quantize = true)
            : base(model_asset, backend_type, apply_quantize)
        {
            Initialize();
        }

        /// <summary>
        /// dispose classification model
        /// </summary>
        public new void Dispose()
        {
            base.Dispose();
        }

        /// <summary>
        /// set apply softmax to confidencies
        /// </summary>
        /// <param name="apply_softmax">apply softmax</param>
        public void SetApplySoftmax(bool apply_softmax = true)
        {
            this.apply_softmax = apply_softmax;
        }

        /// <summary>
        /// classify (get class with highest confidence score)
        /// </summary>
        /// <param name="image">input image</param>
        /// <returns>top-1 class's class-id and confidence score</returns>
        public (int class_id, float score) Classify(Texture2D image)
        {
            var output_tensors = Predict(image);
            var output_name = runtime_model.outputs[0].name;
            var output_tensor = output_tensors[output_name] as TensorFloat;

            var confidences = TensorFloat.AllocNoData(output_tensor.shape);
            if (apply_softmax)
            {
                backend.Softmax(output_tensor, confidences, -1);
            }
            else
            {
                backend.MemCopy(output_tensor, confidences);
            }

            var index = TensorInt.AllocNoData(new TensorShape(1));
            backend.ArgMax(confidences, index, -1, false);

            confidences = confidences.ReadbackAndClone();
            index = index.ReadbackAndClone();

            var class_id = index[0];
            var score = confidences[class_id];

            confidences.Dispose();
            index.Dispose();
            output_tensors.AllDispose();

            return (class_id, score);
        }

        /// <summary>
        /// classify with split predict over multiple frames (get class with highest confidence score)
        /// </summary>
        /// <param name="image">input image</param>
        /// <param name="return_callback">return callback</param>
        /// <returns>callback function to returns top-1 class's class-id and confidence score</returns>
        public IEnumerator Classify(Texture2D image, Action<(int class_id, float score)> return_callback)
        {
            var output_tensors = new Dictionary<string, Tensor>();
            yield return CoroutineHandler.StartStaticCoroutine(Predict(image, (outputs) => output_tensors = outputs));
            var output_name = runtime_model.outputs[0].name;
            var output_tensor = output_tensors[output_name] as TensorFloat;

            var confidences = TensorFloat.AllocNoData(output_tensor.shape);
            if (apply_softmax)
            {
                backend.Softmax(output_tensor, confidences, -1);
            }
            else
            {
                backend.MemCopy(output_tensor, confidences);
            }

            var index = TensorInt.AllocNoData(new TensorShape(1));
            backend.ArgMax(confidences, index, -1, false);

            confidences = confidences.ReadbackAndClone();
            index = index.ReadbackAndClone();

            var class_id = index[0];
            var score = confidences[class_id];

            confidences.Dispose();
            index.Dispose();
            output_tensors.AllDispose();

            return_callback((class_id, score));
        }

        /// <summary>
        /// classify (get top-k classes in order of confidence score)
        /// </summary>
        /// <param name="image">input image</param>
        /// <param name="topk">top-k</param>
        /// <returns>top-k class's class-ids and confidence scores list</returns>
        public List<(int class_id, float score)> Classify(Texture2D image, int topk)
        {
            var output_tensors = Predict(image);
            var output_name = runtime_model.outputs[0].name;
            var output_tensor = output_tensors[output_name] as TensorFloat;

            var confidences = TensorFloat.AllocNoData(output_tensor.shape);
            if (apply_softmax)
            {
                backend.Softmax(output_tensor, confidences, -1);
            }
            else
            {
                backend.MemCopy(output_tensor, confidences);
            }

            var topk_values = TensorFloat.AllocNoData(output_tensor.shape);
            var topk_indices = TensorInt.AllocNoData(output_tensor.shape);
            backend.TopK(confidences, topk_values, topk_indices, topk, -1, true);

            topk_values = topk_values.ReadbackAndClone();
            topk_indices = topk_indices.ReadbackAndClone();

            var classes = new List<(int, float)>(topk);
            foreach (int i in Enumerable.Range(0, topk))
            {
                var class_id = topk_indices[i];
                var score = topk_values[i];
                classes.Add((class_id, score));
            }

            topk_values.Dispose();
            topk_indices.Dispose();
            output_tensors.AllDispose();

            return classes;
        }

        /// <summary>
        /// classify with split predict over multiple frames (get class with highest confidence score)
        /// </summary>
        /// <param name="image">input image</param>
        /// <param name="topk">top-k</param>
        /// <param name="return_callback">return callback</param>
        /// <returns>callback function to returns top-k class's class-ids and confidence scores list</returns>
        public IEnumerator Classify(Texture2D image, int topk, Action<List<(int class_id, float score)>> return_callback)
        {
            var output_tensors = new Dictionary<string, Tensor>();
            yield return CoroutineHandler.StartStaticCoroutine(Predict(image, (outputs) => output_tensors = outputs));
            var output_name = runtime_model.outputs[0].name;
            var output_tensor = output_tensors[output_name] as TensorFloat;

            var confidences = TensorFloat.AllocNoData(output_tensor.shape);
            if (apply_softmax)
            {
                backend.Softmax(output_tensor, confidences, -1);
            }
            else
            {
                backend.MemCopy(output_tensor, confidences);
            }

            var topk_values = TensorFloat.AllocNoData(output_tensor.shape);
            var topk_indices = TensorInt.AllocNoData(output_tensor.shape);
            backend.TopK(confidences, topk_values, topk_indices, topk, -1, true);

            topk_values = topk_values.ReadbackAndClone();
            topk_indices = topk_indices.ReadbackAndClone();

            var classes = new List<(int, float)>(topk);
            foreach (int i in Enumerable.Range(0, topk))
            {
                var class_id = topk_indices[i];
                var score = topk_values[i];
                classes.Add((class_id, score));
            }

            topk_values.Dispose();
            topk_indices.Dispose();
            output_tensors.AllDispose();

            return_callback(classes);
        }

        private void Initialize()
        {
            SetInputMean(new Vector3(0.485f, 0.456f, 0.406f));
            SetInputStd(new Vector3(0.229f, 0.224f, 0.225f));
            SetLayersPerFrame(runtime_model.layers.Count / 5); // TODO : automatic adjust number of layers per frame
        }
    }
}
