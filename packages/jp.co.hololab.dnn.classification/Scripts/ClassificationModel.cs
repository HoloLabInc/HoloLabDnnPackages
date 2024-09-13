using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis;
using Unity.Collections;
using HoloLab.DNN.Base;

namespace HoloLab.DNN.Classification
{
    /// <summary>
    /// classification model class for general classification models
    /// (this class supports models with output shape is 1 x num_classes)
    /// </summary>
    public class ClassificationModel : BaseModel, IDisposable
    {
        /// <summary>
        /// create classification model from sentis file
        /// </summary>
        /// <param name="file_path">model file path</param>
        /// <param name="backend_type">backend type for inference engine</param>
        public ClassificationModel(string file_path, BackendType backend_type = BackendType.GPUCompute)
            : base(file_path, backend_type)
        {
            Initialize();
        }

        /// <summary>
        /// create classification model from stream
        /// </summary>
        /// <param name="stream">model stream</param>
        /// <param name="backend_type">backend type for inference engine</param>
        public ClassificationModel(System.IO.Stream stream, BackendType backend_type = BackendType.GPUCompute)
            : base(stream, backend_type)
        {
            Initialize();
        }

        /// <summary>
        /// create classification model from model asset
        /// </summary>
        /// <param name="model_asset">model asset</param>
        /// <param name="backend_type">backend type for inference engine</param>
        public ClassificationModel(ModelAsset model_asset, BackendType backend_type = BackendType.GPUCompute)
            : base(model_asset, backend_type)
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
        /// apply softmax to confidencies
        /// </summary>
        /// <param name="apply_softmax">apply softmax</param>
        public void ApplySoftmax()
        {
            if (is_quantized)
            {
                throw new Exception("[error] please run this function before weights quantization.");
            }

            SetEditedModel(AddPostProcess());
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
            var output_tensor = output_tensors[output_name] as Tensor<float>;

            var confidences_tensor = output_tensor.ReadbackAndClone();
            var confidences = confidences_tensor.AsReadOnlyNativeArray();

            var score = confidences.Max();
            var class_id = confidences.IndexOf(score);

            confidences_tensor.Dispose();
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
            var output_tensor = output_tensors[output_name] as Tensor<float>;

            var confidences_tensor = output_tensor.ReadbackAndClone();
            var confidences = confidences_tensor.AsReadOnlyNativeArray();

            var score = confidences.Max();
            var class_id = confidences.IndexOf(score);

            confidences_tensor.Dispose();
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
            var output_tensor = output_tensors[output_name] as Tensor<float>;

            var confidences_tensor = output_tensor.ReadbackAndClone();
            var confidences = confidences_tensor.AsReadOnlyNativeArray();

            var sorted_confidences = confidences.Select((confidence, index) => (confidence, index))
                                                .OrderBy(tuple => -tuple.confidence)
                                                .ToArray();

            var classes = new List<(int, float)>(topk);
            foreach (int i in Enumerable.Range(0, topk))
            {
                var score = sorted_confidences[i].confidence;
                var class_id = sorted_confidences[i].index;
                classes.Add((class_id, score));
            }

            confidences_tensor.Dispose();
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
            var output_tensor = output_tensors[output_name] as Tensor<float>;

            var confidences_tensor = output_tensor.ReadbackAndClone();
            var confidences = confidences_tensor.AsReadOnlyNativeArray();

            var sorted_confidences = confidences.Select((confidence, index) => (confidence, index))
                                                .OrderBy(tuple => -tuple.confidence)
                                                .ToArray();

            var classes = new List<(int, float)>(topk);
            foreach (int i in Enumerable.Range(0, topk))
            {
                var score = sorted_confidences[i].confidence;
                var class_id = sorted_confidences[i].index;
                classes.Add((class_id, score));
            }

            confidences_tensor.Dispose();
            output_tensors.AllDispose();

            return_callback(classes);
        }

        private void Initialize()
        {
            SetInputMean(new Vector3(0.485f, 0.456f, 0.406f));
            SetInputStd(new Vector3(0.229f, 0.224f, 0.225f));
            SetSliceFrames(5); // TODO : automatic adjust number of layers per frame
        }

        private Model AddPostProcess()
        {
            try
            {
                var functional_graph = new FunctionalGraph();
                var inputs = functional_graph.AddInputs(base.runtime_model);
                var outputs = Functional.Forward(base.runtime_model, inputs);
                var confidences = Functional.Softmax(outputs[0]);
                var edited_model = functional_graph.Compile(confidences);

                return edited_model;
            }
            catch (Exception e)
            {
                throw new Exception($"[error] can not add post process to model for some reason. ({e.Message})");
            }
        }
    }
}
