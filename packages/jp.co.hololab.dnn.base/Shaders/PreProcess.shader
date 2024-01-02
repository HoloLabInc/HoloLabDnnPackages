Shader "BaseModel/PreProcess"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _Mean("Mean", Vector) = (0.0, 0.0, 0.0, 0.0)
        _Std("Std", Vector) = (1.0, 1.0, 1.0, 0.0)
        _Max("Max", Float) = 1.0
    }
    SubShader
    {
        // No culling or depth
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            sampler2D _MainTex;
            float4 _Mean;
            float4 _Std;
            float _Max;

            float4 frag (v2f i) : SV_Target
            {
                float4 color = tex2D(_MainTex, i.uv);
                color.r = ((color.r - _Mean.r) / _Std.r) * _Max;
                color.g = ((color.g - _Mean.g) / _Std.g) * _Max;
                color.b = ((color.b - _Mean.b) / _Std.b) * _Max;
                return color;
            }
            ENDCG
        }
    }
}
