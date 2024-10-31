using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using UnityEngine;

namespace HoloLab.DNN.ObjectDetection
{
    /// <summary>
    /// non maximum suppression class
    /// </summary>
    public static class NonMaximumSuppression
    {
        /// <summary>
        /// mon maximum suppression
        /// </summary>
        /// <remarks>
        /// objects with confidence score less than specified threshold are supressed.
        /// objects with ratio of overlapping areas more than specified threshold are supressed, and leaving the high confidence score object.
        /// </remarks>
        /// <param name="objects">input objects list</param>
        /// <param name="score_threshold">confidence score threshold [0.0f-1.0f]</param>
        /// <param name="iou_threshold">iou threshold [0.0f-1.0f]</param>
        /// <returns>output object list</returns>
        public static List<HoloLab.DNN.ObjectDetection.BoundingBox> NMS(List<HoloLab.DNN.ObjectDetection.BoundingBox> objects, float score_threshold, float iou_threshold)
        {
            var ordered_objects = objects.Where(o => o.score > score_threshold)
                                         .OrderByDescending(o => o.score)
                                         .ToList();

            var apply_remove = Enumerable.Repeat(false, ordered_objects.Count).ToList();
            for (int i = 0; i < (ordered_objects.Count - 1); i++)
            {
                for (int j = i + 1; j < ordered_objects.Count; j++)
                {
                    if (ordered_objects[i].class_id != ordered_objects[j].class_id)
                    {
                        continue;
                    }

                    var iou = IoU(ordered_objects[i].rect, ordered_objects[j].rect);
                    apply_remove[j] = (iou_threshold < iou);
                }
            }

            var suppressed_objects = new List<HoloLab.DNN.ObjectDetection.BoundingBox>();
            for (int i = 0; i < ordered_objects.Count; i++)
            {
                if (apply_remove[i])
                {
                    continue;
                }

                suppressed_objects.Add(ordered_objects[i]);
            }

            return suppressed_objects;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float IoU(Rect rect1, Rect rect2)
        {
            if (!rect1.Overlaps(rect2))
            {
                return 0.0f;
            }

            var itl = new Vector2(
                Math.Max(rect1.min.x, rect2.min.x),
                Math.Max(rect1.min.y, rect2.min.y)
            );
            var ibr = new Vector2(
                Math.Min(rect1.max.x, rect2.max.x),
                Math.Min(rect1.max.y, rect2.max.y)
            );

            float intersection = (ibr.x - itl.x) * (ibr.y - itl.y);
            float rect1_area = rect1.width * rect1.height;
            float rect2_area = rect2.width * rect2.height;
            float union = (rect1_area + rect2_area - intersection);

            return intersection / union;
        }
    }
}
