using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis;

namespace HoloLab.DNN.Base
{
    /// <summary>
    /// enumerable extension methods class
    /// </summary>
    public static class EnumerableExtensions
    {
        /// <summary>
        /// dispose all tensors for tensor
        /// </summary>
        /// <param name="tensors">tensors</param>
        public static void AllDispose<T>(this IEnumerable<T> tensors)
            where T : Tensor
        {
            foreach (var tensor in tensors)
            {
                tensor?.Dispose();
            }
        }
    }
}
