using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using UnityEngine;
using UnityEngine.UI;
using Unity.Sentis;
using HoloLab.DNN.Classification;

namespace Sample
{
    public class Classification : MonoBehaviour
    {
        [SerializeField, Tooltip("Input Image")] private RawImage input_image = null;
        [SerializeField, Tooltip("Weights")] private ModelAsset weights = null;
        [SerializeField, Tooltip("Label List")] private TextAsset names = null;
        [SerializeField, Tooltip("Result Text")] private Text text = null;
        [SerializeField, Tooltip("Apply Softmax")] private bool apply_softmax = true;
        [SerializeField, Tooltip("Mean")] private Vector3 mean = new Vector3(0.485f, 0.456f, 0.406f);
        [SerializeField, Tooltip("Std")] private Vector3 std = new Vector3(0.229f, 0.224f, 0.225f);

        private HoloLab.DNN.Classification.ClassificationModel model;
        private List<string> labels;

        private void Start()
        {
            // Create Classification Model
            model = new HoloLab.DNN.Classification.ClassificationModel(weights);
            if (apply_softmax) { model.ApplySoftmax(); }
            model.ApplyQuantize();
            model.SetInputMean(mean);
            model.SetInputStd(std);

            // Read Label List from Text Asset
            labels = new List<string>(Regex.Split(names.text, "\r\n|\r|\n"));
        }

        public void OnClick()
        {
            // Get Texture from Raw Image
            var input_texture = input_image.texture as Texture2D;
            if (input_texture == null)
            {
                return;
            }

            // Crop Texture from Center
            var croped_texture = HoloLab.DNN.Classification.Crop.CenterCrop(input_texture);

            // Classify
            (var class_id, var score) = model.Classify(croped_texture);

            // Show Class on Unity Console
            Debug.Log($"{class_id} {labels[class_id]} ({score:F3})");
            
            // Show Class on Unity UI
            text.text = $"{class_id} {labels[class_id]} ({score:F3})";

            // Destroy Texture
            Destroy(croped_texture);
        }

        private void OnDestroy()
        {
            model?.Dispose();
            model = null;
        }
    }
}
