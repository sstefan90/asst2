import texture;

public float4 sampling_fwd(
    uint2 tid,
    uint2 canvasSize,
    StructuredBuffer<float4> sampleTex,
    uint2 texSize)
{

    float2 uv = (float2(tid) + 0.5) / float2(canvasSize);
    DiffTexture2D<float, 4> diffTex = DiffTexture2D<float, 4>(sampleTex, texSize);

    return sampleBilinear(diffTex,  uv);

}

public void sampling_bwd(
    uint2 tid,
    uint2 canvasSize,
    StructuredBuffer<float4> sampleTex,
    RWStructuredBuffer<Atomic<float>[4]> sampleTexGrad,
    uint2 texSize,
    float4 dOut)
{
    float2 uv = (float2(tid) + 0.5) / float2(canvasSize);
    DiffTexture2D_Differential<float, 4> sampleTexGrad_ = DiffTexture2D_Differential<float, 4>(sampleTexGrad);
    DiffTexture2D<float, 4> diffTex = DiffTexture2D<float, 4>(sampleTex, texSize);
    DifferentialPtrPair<DiffTexture2D<float, 4>> diffTex_d = DifferentialPtrPair<DiffTexture2D<float, 4>>(diffTex, sampleTexGrad_);
    return bwd_diff(sampleBilinear)(diffTex_d, uv, dOut);
}
