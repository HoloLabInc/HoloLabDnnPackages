#if UNITY_EDITOR
using System.Linq;
using UnityEngine;
using UnityEditor;

namespace HoloLab.DNN.Base
{
    /// <summary>
    /// editor extension class
    /// </summary>
    public static class EditorExtension
    {
        /// <summary>
        /// call method on project loaded in editor
        /// </summary>
        [InitializeOnLoadMethod]
        public static void OnProjectLoadedInEditor()
        {
            var shader = Shader.Find("BaseModel/PreProcess");

            if (ExistsInAlwaysIncludedShaders(shader))
            {
                return;
            }

            AddAlwaysIncludedShaders(shader);
        }

        /// <summary>
        /// check exists shader in always included shaders
        /// </summary>
        /// <param name="shader">shader</param>
        /// <returns>if exists shader in always included shaders return true, otherwise, return false</returns>
        public static bool ExistsInAlwaysIncludedShaders(Shader shader)
        {
            var path = "ProjectSettings/GraphicsSettings.asset";
            var manager = AssetDatabase.LoadAllAssetsAtPath(path).FirstOrDefault();

            var serialized_object = new SerializedObject(manager);
            serialized_object.Update();

            var property = serialized_object.FindProperty("m_AlwaysIncludedShaders");
            var size = property.arraySize;

            var exists = false;
            property.NextVisible(true);
            for (var i = 0; i < size; i++)
            {
                property.NextVisible(false);

                if (property.objectReferenceValue == null)
                {
                    continue;
                }

                if (property.objectReferenceValue.Equals(shader))
                {
                    exists = true;
                    break;
                }
            }

            return exists;
        }

        /// <summary>
        /// add shader to always included shaders
        /// </summary>
        /// <param name="shader">shader</param>
        public static void AddAlwaysIncludedShaders(Shader shader)
        {
            var path = "ProjectSettings/GraphicsSettings.asset";
            var manager = AssetDatabase.LoadAllAssetsAtPath(path).FirstOrDefault();

            var serialized_object = new SerializedObject(manager);
            serialized_object.Update();

            var property = serialized_object.FindProperty("m_AlwaysIncludedShaders");
            property.arraySize += 1;
            property.GetArrayElementAtIndex(property.arraySize - 1).objectReferenceValue = shader;

            serialized_object.ApplyModifiedProperties();
        }
    }
}
#endif
