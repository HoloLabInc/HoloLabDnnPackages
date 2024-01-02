using System.Linq;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace HoloLab.DNN.ObjectDetection
{
    /// <summary>
    /// visualization class for objects
    /// </summary>
    public static class Visualizer
    {
        /// <summary>
        /// draw bounding box
        /// </summary>
        /// <param name="graphic">graphical unity ui object</param>
        /// <param name="rect">rect</param>
        /// <param name="color">color</param>
        public static void DrawBoudingBox(Graphic graphic, Rect rect, Color color)
        {
            var texture_size = new Vector2(graphic.mainTexture.width, graphic.mainTexture.height);
            var ui_size = graphic.GetComponent<RectTransform>().sizeDelta;
            var ui_pivot = graphic.GetComponent<RectTransform>().pivot;

            var panel = new GameObject("BoundingBox");
            panel.AddComponent<CanvasRenderer>();
            panel.AddComponent<Image>().color = color;
            panel.transform.SetParent(graphic.transform, false);
            panel.transform.localPosition = new Vector3(
                ((rect.center.x / texture_size.x) - ui_pivot.x) * ui_size.x,
                -((rect.center.y / texture_size.y) - (1.0f - ui_pivot.y)) * ui_size.y
            );
            var rect_transform = panel.GetComponent<RectTransform>();
            rect_transform.sizeDelta = new Vector2(
                (rect.width / texture_size.x) * ui_size.x,
                (rect.height / texture_size.y) * ui_size.y
            );
        }

        /// <summary>
        /// draw label on bounding box
        /// </summary>
        /// <param name="graphic">graphical unity ui object</param>
        /// <param name="rect">rect</param>
        /// <param name="color">color</param>
        /// <param name="label">label</param>
        /// <param name="font">font</param>
        /// <param name="font_size">font size</param>
        public static void DrawLabel(Graphic graphic, Rect rect, Color color, string label, Font font, int font_size = 22)
        {
            var texture_size = new Vector2(graphic.mainTexture.width, graphic.mainTexture.height);
            var ui_size = graphic.GetComponent<RectTransform>().sizeDelta;
            var ui_pivot = graphic.GetComponent<RectTransform>().pivot;

            var game_object = new GameObject("Label");
            var text = game_object.AddComponent<Text>();
            text.text = label;
            text.color = color;
            text.font = font;
            text.fontSize = font_size;
            text.transform.SetParent(graphic.transform, false);
            text.transform.localPosition = new Vector3(
                ((rect.center.x / texture_size.x) - ui_pivot.x) * ui_size.x,
                -((rect.center.y / texture_size.y) - (1.0f - ui_pivot.y)) * ui_size.y
            );
            var rect_transform = game_object.GetComponent<RectTransform>();
            rect_transform.sizeDelta = new Vector2(
                (rect.width / texture_size.x) * ui_size.x,
                (rect.height / texture_size.y) * ui_size.y
            );
        }

        /// <summary>
        /// clear all bounding boxes
        /// </summary>
        /// <param name="graphic">graphical unity ui object</param>
        public static void ClearBoundingBoxes(Graphic graphic)
        {
            foreach (var child in graphic.transform.Cast<Transform>())
            {
                if (!"BoundingBox".Contains(child.name))
                {
                    continue;
                }

                MonoBehaviour.Destroy(child.gameObject);
            }
        }

        /// <summary>
        /// clear all labels
        /// </summary>
        /// <param name="graphic">graphical unity ui object</param>
        public static void ClearLabels(Graphic graphic)
        {
            foreach (var child in graphic.transform.Cast<Transform>())
            {
                if (!"Label".Contains(child.name))
                {
                    continue;
                }

                MonoBehaviour.Destroy(child.gameObject);
            }
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
