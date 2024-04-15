using System.Collections;
using UnityEngine;

namespace HoloLab.DNN.Base
{
    public class CoroutineHandler : MonoBehaviour
    {
        static protected CoroutineHandler _instance;

        static public CoroutineHandler instance
        {
            get
            {
                if (_instance == null)
                {
                    var game_object = new GameObject("CoroutineHandler");
                    DontDestroyOnLoad(game_object);
                    _instance = game_object.AddComponent<CoroutineHandler>();
                }

                return _instance;
            }
        }

        public void OnDisable()
        {
            if (_instance != null)
            {
                Destroy(_instance.gameObject);
                _instance = null;
            }
        }

        static public Coroutine StartStaticCoroutine(IEnumerator coroutine)
        {
            return instance.StartCoroutine(coroutine);
        }
    }
}
