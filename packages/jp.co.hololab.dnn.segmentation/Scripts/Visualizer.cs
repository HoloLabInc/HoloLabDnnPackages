using System.Linq;
using System.Threading.Tasks;
using System.Collections.Generic;
using UnityEngine;

namespace HoloLab.DNN.Segmentation
{
    /// <summary>
    /// visualization class for segmented area
    /// </summary>
    public static class Visualizer
    {
        /// <summary>
        /// colorize area from indices
        /// </summary>
        /// <param name="indices_texture">indices texture with index in color.r</param>
        /// <param name="colors">color list for area indices</param>
        /// <returns>colorized area texture</returns>
        public static Texture2D ColorizeArea(Texture2D indices_texture, List<Color> colors)
        {
            var indices = indices_texture.GetPixels32(0);
            var pixels = new Color[indices_texture.width * indices_texture.height];
            Parallel.For(0, pixels.Length, i => { pixels[i] = colors[indices[i].r]; });
            var colorized_texture = new Texture2D(indices_texture.width, indices_texture.height, TextureFormat.RGBA32, false);
            colorized_texture.SetPixels(pixels);
            colorized_texture.Apply();
            return colorized_texture;
        }

        /// <summary>
        /// generate random colors
        /// </summary>
        /// <param name="num">number of colors</param>
        /// <param name="alpha">alpha value</param>
        /// <param name="seed">random seed value</param>
        /// <returns>generated color list</returns>
        public static List<Color> GenerateRandomColors(int num, float alpha = 0.5f, int seed = 1000)
        {
            UnityEngine.Random.InitState(seed);

            var colors = new List<Color>(num);
            Enumerable.Range(0, num).ToList().ForEach(i =>
                colors.Add(
                    new Color(
                        UnityEngine.Random.value,
                        UnityEngine.Random.value,
                        UnityEngine.Random.value,
                        alpha
                    )
                )
            );

            return colors;
        }
    }
}
