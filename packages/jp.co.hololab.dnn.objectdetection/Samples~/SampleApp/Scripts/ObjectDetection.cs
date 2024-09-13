using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using UnityEngine;
using UnityEngine.UI;
using Unity.Sentis;
using HoloLab.DNN.ObjectDetection;

namespace Sample
{
    public class ObjectDetection : MonoBehaviour
    {
        [SerializeField, Tooltip("Input Image")] private RawImage input_image = null;
        [SerializeField, Tooltip("Weights")] private ModelAsset weights = null;
        [SerializeField, Tooltip("Label List")] private TextAsset names = null;
        [SerializeField, Tooltip("Confidence Score Threshold"), Range(0.0f, 1.0f)] private float score_threshold = 0.6f;
        [SerializeField, Tooltip("IoU Threshold"), Range(0.0f, 1.0f)] private float iou_threshold = 0.4f;

        //private HoloLab.DNN.ObjectDetection.ObjectDetectionModel_YOLOX model;
        private HoloLab.DNN.ObjectDetection.ObjectDetectionModel_YOLOv9 model;
        private Font font;
        private Color color_offset;
        private List<Color> colors;
        private List<string> labels;

        private void Start()
        {
            // Create Object Detection Model
            //model = new HoloLab.DNN.ObjectDetection.ObjectDetectionModel_YOLOX(weights);
            model = new HoloLab.DNN.ObjectDetection.ObjectDetectionModel_YOLOv9(weights);
            model.ApplyQuantize();

            // Read Label List from Text Asset
            labels = new List<string>(Regex.Split(names.text, "\r\n|\r|\n"));

            // Get Font and Create Colors for Visualize
            try
            {
                font = Resources.GetBuiltinResource(typeof(Font), "Arial.ttf") as Font;
            }
            catch
            {
                font = Resources.GetBuiltinResource(typeof(Font), "LegacyRuntime.ttf") as Font;
            }
            colors = HoloLab.DNN.ObjectDetection.Visualizer.GenerateRandomColors(labels.Count);
            color_offset = new Color(0.5f, 0.5f, 0.5f, 0.0f);
        }

        public void OnClick()
        {
            // Get Texture from Raw Image
            var input_texture = input_image.texture as Texture2D;
            if (input_texture == null)
            {
                return;
            }

            // Detect Objects
            var objects = model.Detect(input_texture, score_threshold, iou_threshold);

            // Show Objects on Unity Console
            objects.ForEach(o => Debug.Log($"{o.class_id} {labels[o.class_id]} ({o.score:F2}) : {o.rect}"));

            // Draw Objects on Unity UI
            HoloLab.DNN.ObjectDetection.Visualizer.ClearBoundingBoxes(input_image);
            objects.ForEach(o => HoloLab.DNN.ObjectDetection.Visualizer.DrawBoudingBox(input_image, o.rect, colors[o.class_id]));

            HoloLab.DNN.ObjectDetection.Visualizer.ClearLabels(input_image);
            objects.ForEach(o => HoloLab.DNN.ObjectDetection.Visualizer.DrawLabel(input_image, o.rect, colors[o.class_id] - color_offset, $"{labels[o.class_id]} ({o.score:F2})", font));
        }
        private void OnDestroy()
        {
            model?.Dispose();
            model = null;
        }
    }
}
