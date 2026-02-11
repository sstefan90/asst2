implementing "../renderer.slang";

import math;
import texture;

public static uint MAX_STEPS = 1024;
public static float EPSILON = 1e-6;

public static float STEP_SIZE = 0.01;
public static float DENSITY_SCALE = 5.0;

/**
 * Perform ray marching through the volume and accumulate color.
 * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @param tMax The maximum t value for ray marching (usually the nearest primitive intersection t).
 * @param primColor The color of the nearest primitive intersection.
 * @return The accumulated color after ray marching through the volume.
 */
public float4 volumeSample(float2 uv, RendererUniform uniforms, float tMax, float4 primColor)
{
    
    
    Ray ray = uniforms.camera.generateRay(uv);
    ray.origin = mul(uniforms.volume.invModelMatrix, float4(ray.origin, 1.0)).xyz;
    ray.direction = mul(uniforms.volume.invModelMatrix, float4(ray.direction, 0.0)).xyz;
    float dirscale = length(ray.direction);
    ray.direction = normalize(ray.direction);
    BoundingBox volumeBound = uniforms.volume.bound;
    float2 t_data = volumeBound.hit(ray);
    float t_enter = t_data[0];
    float t_exit = t_data[1];
    tMax = tMax * dirscale;


    if (t_enter < 0 || t_exit < 0) {
        return primColor;
    }

    if (tMax < t_enter) {
        //primitive in front of volume
        return primColor;
    }

    float t_end = min(t_exit, tMax);

    if (t_end < t_enter) {
        return primColor;
    }

    float stepSize = STEP_SIZE;

    uint neededSteps = uint(ceil((t_end - t_enter) / STEP_SIZE));

    if (neededSteps > MAX_STEPS) {
        //increase step size so we can make it across
        stepSize = (t_end - t_enter) / float(MAX_STEPS);
    }

    uint numSteps = min(MAX_STEPS, uint(ceil((t_end - t_enter) / STEP_SIZE)));

    //if (stepSize > STEP_SIZE){
    //    return float4(1.0, 0.0, 0.0, 1.0); //red
    //}
    
    return accumulateColor<4>(ray, volumeBound, numSteps, stepSize, DENSITY_SCALE, primColor, uniforms.volume.tex);
    
    // TODO: Student implementation ends here.
}

/**
 * Backward pass for volumetric ray marching at a single uv.
 * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @param outGrad The output gradient w.r.t. the sampled color.
 */
public void volumeSampleBwd(float2 uv, RendererUniform uniforms, float4 outGrad)
{
    // TODO: Student implementation starts here.

    // TODO: Student implementation ends here.
}

/**
 * Ray march through the volume and accumulate color.
 * @param ray The ray in local space.
 * @param volumeBound The bounding box of the volume in local space.
 * @param maxSteps The maximum number of marching steps.
 * @param stepSize The size of each marching step.
 * @param densityScale The scale factor for the volume density (alpha channel).
 * @param backgroundColor The background color behind the volume
                          (this can be the ambient color if there's no primitive behind the volume, or the color of the primitive hit if there's one).
 * @param data The 3D texture storing the volume data (rgba).
 * @return The accumulated color after ray marching through the volume.
 */
[Differentiable]
public float4 accumulateColor<let N : uint>(
    no_diff Ray ray,
    no_diff BoundingBox volumeBound,
    no_diff uint maxSteps,
    no_diff float stepSize,
    no_diff float densityScale,
    no_diff float4 backgroundColor,
    DiffTexture3D<float, 4> data)
{
    
    float4 color = backgroundColor;
    float start_t = volumeBound.hit(ray)[0];
    float3 v_min = volumeBound.pMin;
    float3 v_max = volumeBound.pMax;
    static const uint MAX_STEPS_ = 1024;
    float alpha = 0.0;
    float totalDensity = 0.0;
    
    [ForceUnroll]
    for (uint i = 0; i < MAX_STEPS_; i++) {
        if (i < maxSteps) {
            
        
        float t = start_t + (float(i) + 0.5) * stepSize;
        float3 cur_point = ray.origin + t * ray.direction;

        float3 uv = (cur_point - v_min) / (v_max - v_min);
        uv = clamp(uv, float3(0.0), float3(1.0));
        float4 sample = sampleTrilinear(data, uv);
        sample = max(sample, float4(0.0));

        alpha = max(sample.a * stepSize * densityScale, 0.0);
        color.rgb = color.rgb * (1.0 - alpha) + sample.rgb * alpha;
        }
        

    }
    color = clamp(color, float4(0.0), float4(1.0));
    return color;
}

