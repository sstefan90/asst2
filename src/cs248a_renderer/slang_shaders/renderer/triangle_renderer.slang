implementing "../renderer.slang";


float3 simpleShade(float3 rayDir, float3 normal, float3 albedo)
{
    float NdotL = max(dot(normalize(normal), normalize(-rayDir)), 0.0);
    return albedo * NdotL;
}

public struct RayMeshSampleResult
{
    public float t;
    public float3 normal;
    public float4 color;
}

public struct RayMeshIntersectionResult
{
    public RayHitResult closestHit;
    public Triangle hitTriangle;
}

public Optional<RayMeshSampleResult> meshSample(float2 uv, RendererUniform uniforms)
{
    // We generate a camera ray for the given uv coordinate.
    // You need to implement the `generateRay` function in `Camera` struct in `src/cs248a_renderer/slang_shaders/model/camera.slang`
    Ray ray = uniforms.camera.generateRay(uv);
    RayHitResult result;

    Optional<RayMeshSampleResult> allResult;
    // Ray mesh intersection test.
    Optional<RayMeshIntersectionResult> rayMeshIntersectionResult = rayMeshIntersection(ray, uniforms);
    if (rayMeshIntersectionResult.hasValue)
    {
        RayHitResult result = rayMeshIntersectionResult.value.closestHit;
        float3 hitPoint = ray.origin + result.t * ray.direction;
        float3 barycentricCoord = rayMeshIntersectionResult.value.hitTriangle.calculateBarycentricCoord(hitPoint);
        float3 normal = result.normal;
        float2 tex_uv = rayMeshIntersectionResult.value.hitTriangle.getUV(barycentricCoord);
        float4 color = shadeTriangle(uv, uniforms, rayMeshIntersectionResult.value.hitTriangle, ray, tex_uv, normal, barycentricCoord);
        allResult = Optional<RayMeshSampleResult>(RayMeshSampleResult(result.t, normal, color));
    }

    return allResult;
}

public Optional<RayMeshIntersectionResult> rayTriangles(Ray ray, StructuredBuffer<Triangle> triangles, uint triangleCount)
{

    // find the normal of the plane defined by the triangle
    Optional<RayHitResult> result;
    Triangle hitTriangle;
    float t_smallest = float.maxValue;
    for (int i = 0; i < triangleCount; i++)
    {
        Optional<RayHitResult> hitResult = triangles[i].hit(ray);
        if (hitResult.hasValue) {
            if (hitResult.value.t < t_smallest)
            {
                t_smallest = hitResult.value.t;
                result = hitResult;
                hitTriangle = triangles[i];
            }
        }
    }

    if (t_smallest == float.maxValue)
    {
        return Optional<RayMeshIntersectionResult>();
    }

    return Optional<RayMeshIntersectionResult>(RayMeshIntersectionResult(result.value, hitTriangle));

}

public Optional<RayMeshIntersectionResult> rayMeshIntersection(Ray ray, RendererUniform uniforms)
{
    Optional<RayHitResult> result = Optional<RayHitResult>();

    // IMPORTANT CHANGE: We need to store the hit triangle.
    Optional<RayMeshIntersectionResult> allResult = Optional<RayMeshIntersectionResult>();
    //Optional<Triangle> hitTriangle = Optional<Triangle>();
    // IMPORTANT CHANGE: We need to store the hit triangle.

    // TODO: Student implementation starts here.
    float t_min = float.maxValue;

    if (!uniforms.useBVH) {
    
        allResult = rayTriangles(ray, uniforms.triangleBuf, uniforms.triangleCount);
    }
    else {

        Optional<uniforms.bvh.HitResult> res = uniforms.bvh.hit(ray);

        if (res.hasValue) {
            Triangle hitTriangle = res.value.primitive;
            RayHitResult rayHitResult = res.value.rayHitResult;
            allResult = Optional<RayMeshIntersectionResult>(RayMeshIntersectionResult(rayHitResult, hitTriangle));

        }

    }
    
    // TODO: Student implementation ends here.

    //{
    //    allResult = Optional<RayMeshIntersectionResult>(RayMeshIntersectionResult(result.value, hitTriangle.value));
    //}

    return allResult;
}

// Compute the level of the texture at the hit point.
// @param uv: The uv coordinates of the hit point.
// @param uniforms: The renderer uniforms.
// @param triangle: The triangle that was hit.
// @param tex_uv: The uv coordinates of the texture at the hit point.
// @return: The level of the texture at the hit point.
public float getLevel(float2 uv, RendererUniform uniforms, Triangle triangle, float2 tex_uv)
{
    

    float delta = 1e-6;

    //launch 4 rays to compute the level of detail
    // + dx
    Ray ray_dx = uniforms.camera.generateRay(uv + float2(delta, 0.0));
    // - dx
    Ray ray_minus_dx = uniforms.camera.generateRay(uv - float2(delta, 0.0));
    // + dy
    Ray ray_dy = uniforms.camera.generateRay(uv + float2(0.0, delta));
    // - dy
    Ray ray_minus_dy = uniforms.camera.generateRay(uv - float2(0.0, delta));

    Optional<RayHitResult> res_dx = triangle.hit(ray_dx);
    Optional<RayHitResult> res_minus_dx = triangle.hit(ray_minus_dx);
    Optional<RayHitResult> res_dy = triangle.hit(ray_dy);
    Optional<RayHitResult> res_minus_dy = triangle.hit(ray_minus_dy);

    float2 uv_dx = getTextCoord(triangle, ray_dx, res_dx, tex_uv);
    float2 uv_minus_dx = getTextCoord(triangle, ray_minus_dx, res_minus_dx, tex_uv);
    float2 uv_dy = getTextCoord(triangle, ray_dy, res_dy, tex_uv);
    float2 uv_minus_dy = getTextCoord(triangle, ray_minus_dy, res_minus_dy, tex_uv);

    float2 duv_dx = (uv_dx - uv_minus_dx) / (2.0*delta);
    float2 duv_dy = (uv_dy - uv_minus_dy) / (2.0*delta);

    float px = length(duv_dx);
    float py = length(duv_dy);
    float level = log2(max(px, py));


    return level;
}

float2 getTextCoord(Triangle triangle, Ray ray, Optional<RayHitResult> hitResult, float2 tex_uv){
    if (!hitResult.hasValue) {
        return tex_uv;
    }

    float3 hitPoint = ray.origin + hitResult.value.t * ray.direction;
    float3 barycentricCoord = triangle.calculateBarycentricCoord(hitPoint);
    float2 tex_uv = triangle.getUV(barycentricCoord);
    return tex_uv;
}

public float4 shadeTriangle(float2 uv, RendererUniform uniforms, Triangle triangle, Ray ray, float2 tex_uv, float3 normal, float3 barycentricCoord)
{
    if (uniforms.visualizeBarycentricCoords)
    {
        return float4(barycentricCoord, 1.0);
    }
    if (uniforms.visualizeTexUV)
    {
        return float4(tex_uv, 0.0, 1.0);
    }
    // Get the level of detail of the texture at the hit point.
    float level = getLevel(uv, uniforms, triangle, tex_uv);
    uint discreteLevel = uint(level);

    // Sample albedo from the material at the hit point.
    uint materialId = triangle.materialId + triangle.offset;
    PhysicsBasedMaterial physicsBasedMaterial = uniforms.physicsBasedMaterialBuf[materialId];
    PhysicsBasedParameters parameters = sampleMaterial(physicsBasedMaterial, uniforms.physicsBasedMaterialTextureBuf, tex_uv, level, uniforms.visualizeLevelOfDetail);
    float3 albedo = parameters.albedo;

    if (uniforms.visualizeLevelOfDetail || uniforms.visualizeAlbedo)
    {
        return float4(float3(albedo), 1.0);
    }

    // Shade the triangle.
    float3 color = simpleShade(ray.direction, normal, albedo);
    return float4(color, 1.0);
}
