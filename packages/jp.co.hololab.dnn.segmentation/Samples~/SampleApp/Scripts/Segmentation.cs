using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Unity.Sentis;
using HoloLab.DNN.Segmentation;

namespace Sample
{
    public class Segmentation : MonoBehaviour
    {
        [SerializeField, Tooltip("Input Image")] private RawImage input_image = null;
        [SerializeField, Tooltip("Output Image")] private RawImage output_image = null;
        [SerializeField, Tooltip("Weights")] private ModelAsset weights = null;
        [SerializeField, Tooltip("Alpha"), Range(0.0f, 1.0f)] private float alpha = 0.75f;
        [SerializeField, Tooltip("Mean")] private Vector3 mean = new Vector3(0.485f, 0.456f, 0.406f);
        [SerializeField, Tooltip("Std")] private Vector3 std = new Vector3(0.229f, 0.224f, 0.225f);

        private HoloLab.DNN.Segmentation.SegmentationModel model = null;
        private List<Color> colors;

        private void Start()
        {
            // Create Segmentation Model
            model = new HoloLab.DNN.Segmentation.SegmentationModel(weights);
            model.ApplyQuantize();
            model.SetInputMean(mean);
            model.SetInputStd(std);

            // Create Colors for Visualize
            var num_classes = model.GetNumClasses();
            var random_seed = 0;
            colors = HoloLab.DNN.Segmentation.Visualizer.GenerateRandomColors(num_classes, alpha, random_seed);
            colors[0] = Color.clear; // index 0 is background area in sample model
        }

        public void OnClick()
        {
            // Get Texture from Raw Image
            var input_texture = input_image.texture as Texture2D;
            if (input_texture == null)
            {
                return;
            }

            // Segment Area
            var indices_texture = model.Segment(input_texture);

            // Draw Area on Unity UI
            var colorized_texture = HoloLab.DNN.Segmentation.Visualizer.ColorizeArea(indices_texture, colors);
            if (output_image.texture == null)
            {
                output_image.color = Color.white;
                output_image.texture = new Texture2D(indices_texture.width, indices_texture.height, TextureFormat.RGBA32, false);
            }
            Graphics.CopyTexture(colorized_texture, output_image.texture);

            // Destroy Texture
            Destroy(colorized_texture);
            Destroy(indices_texture);
        }

        private void OnDestroy()
        {
            model?.Dispose();
            model = null;
        }
    }
}
