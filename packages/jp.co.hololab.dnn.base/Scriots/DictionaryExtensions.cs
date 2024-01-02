using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis;

namespace HoloLab.DNN.Base
{
    /// <summary>
    /// dictionary extension methods class
    /// </summary>
    public static class DictionaryExtensions
    {
        /// <summary>
        /// dispose all tensors for output tensors from predict
        /// </summary>
        /// <param name="output_tensors">output tensors</param>
        public static void AllDispose(this Dictionary<string, Tensor> output_tensors)
        {
            foreach (var output_tensor in output_tensors)
            {
                output_tensor.Value?.Dispose();
            }
        }
    }
}
