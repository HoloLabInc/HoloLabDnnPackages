using UnityEngine;
using UnityEngine.UI;
using Unity.Sentis;
using HoloLab.DNN.DepthEstimation;

namespace Sample
{
    public class DepthEstimation : MonoBehaviour
    {
        [SerializeField, Tooltip("Input Image")] private RawImage input_image = null;
        [SerializeField, Tooltip("Output Image")] private RawImage output_image = null;
        [SerializeField, Tooltip("Weights")] private ModelAsset weights = null;
        [SerializeField, Tooltip("Mean")] private Vector3 mean = new Vector3(0.5f, 0.5f, 0.5f);
        [SerializeField, Tooltip("Std")] private Vector3 std = new Vector3(0.5f, 0.5f, 0.5f);

        private HoloLab.DNN.DepthEstimation.DepthEstimationModel model;

        private void Start()
        {
            // Create Depth Estimation Model
            model = new HoloLab.DNN.DepthEstimation.DepthEstimationModel(weights);
            model.SetInputMean(mean);
            model.SetInputStd(std);
        }

        public void OnClick()
        {
            // Get Texture from Raw Image
            var input_texture = input_image.texture as Texture2D;
            if (input_texture == null)
            {
                return;
            }

            // Estimate Depth
            var depth_texture = model.Estimate(input_texture);

            // Draw Depth on Unity UI
            if (output_image.texture == null)
            {
                output_image.texture = new Texture2D(depth_texture.width, depth_texture.height, depth_texture.format, false);
            }
            Graphics.CopyTexture(depth_texture, output_image.texture);

            // Destroy Texture
            Destroy(depth_texture);
        }

        private void OnDestroy()
        {
            model?.Dispose();
            model = null;
        }
    }
}
