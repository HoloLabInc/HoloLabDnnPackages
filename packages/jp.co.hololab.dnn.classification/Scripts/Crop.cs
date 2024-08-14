using System;
using UnityEngine;

namespace HoloLab.DNN.Classification
{
    /// <summary>
    /// image crop class
    /// </summary>
    public static class Crop
    {
        /// <summary>
        /// square crop image from center
        /// </summary>
        /// <param name="image">input image</param>
        /// <returns>center croped image</returns>
        public static Texture2D CenterCrop(Texture2D image)
        {
            var short_side = Math.Min(image.width, image.height);
            var result = new Texture2D(short_side, short_side, image.format, false);
            var pixels = image.GetPixels(
                (int)((image.width - short_side) * 0.5f),
                (int)((image.height - short_side) * 0.5f),
                short_side,
                short_side
            );
            result.SetPixels(pixels);
            result.Apply(false);

            return result;
        }
    }
}
