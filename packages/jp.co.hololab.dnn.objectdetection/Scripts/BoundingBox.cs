using System;
using UnityEngine;

namespace HoloLab.DNN.ObjectDetection
{
    /// <summary>
    /// bounding box struct
    /// </summary>
    [Serializable]
    public struct BoundingBox
    {
        /// <value>
        /// rectangle
        /// </value>
        /// <remarks>
        /// rect.x and rect.y is upper-left corner of bounding box, rect.widht and rect.height is width and height of bounding box.
        /// </remarks>
        public Rect rect { get; set; }

        /// <value>
        /// class-id
        /// </value>
        /// <remarks>
        /// class-id is integer value in range of [0-num_classes).
        /// </remarks>
        public int class_id { get; set; }

        /// <value>
        /// confidence score
        /// </value>
        /// <remarks>
        /// confidence score is floating point number value in range of [0.0f-1.0f].
        /// </remarks>
        public float score { get; set; }

        public BoundingBox(Rect rect, int class_id, float score)
        {
            this.rect = rect;
            this.class_id = class_id;
            this.score = score;
        }
    }
}
