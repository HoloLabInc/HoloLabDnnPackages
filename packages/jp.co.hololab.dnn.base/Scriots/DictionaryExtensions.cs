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
        /// dispose all tensors
        /// </summary>
        /// <param name="tensors">tensors</param>
        public static void AllDispose<T>(this Dictionary<string, T> tensors)
            where T : Tensor
        {
            foreach (var tensor in tensors)
            {
                tensor.Value?.Dispose();
            }
        }
    }
}
