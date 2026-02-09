import math;
import model;
import primitive;

/**
 * Fill a solid red color.
 * @param tid The 2D coordinate of the current thread. Ranges from (0, 0) to (canvasSize.x - 1, canvasSize.y - 1).
 * @return The color red as a float4 vector.
 */
public float4 fillRed(uint2 tid)
{
    // Return the color as a float4 vector (red, green, blue, alpha).
    // In this example, we set alpha to 1 to indicate a fully opaque pixel.
    // We'll cover alpha, transparency later in the class.
    return float4(1.0, 0.0, 0.0, 1.0);
}

/**
 * Fill a uv gradient pattern.
 * @param tid The 2D coordinate of the current thread. Ranges from (0, 0) to (canvasSize.x - 1, canvasSize.y - 1).
 * @param canvasSize The size of the output image in pixels.
 * @return The color corresponding to the uv(xy) coordinate of the pixel.
 */
public float4 fillGradient(uint2 tid, uint2 canvasSize)
{
    float2 result = float2(tid) / float2(canvasSize);


    return float4(result, 0, 1.0);
}

/**
 * Fill a checkerboard pattern.
 * @param tid The 2D coordinate of the current thread. Ranges from (0, 0) to (canvasSize.x - 1, canvasSize.y - 1).
 * @param canvasSize The size of the output image in pixels.
 * @param numSquares The number of squares along each axis.
 * @return The color white or black depending on the checkerboard pattern.
 */
public float4 fillCheckerboard(uint2 tid, uint2 canvasSize, uint numSquares)
{
    float2 norm = float2(tid) / float2(canvasSize);
    uint2 square_sizes = canvasSize / numSquares;

    uint2 square_index = tid  / square_sizes;

    if ((square_index.x + square_index.y) % 2 == 0)
    {
        return float4(1.0, 1.0, 1.0, 1.0);
    }
    else
    {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    // TODO: Student implementation ends here.
}



public uint side_test(float2 p, float2 a, float2 b, float2 c)
{

    float2x2 m = float2x2(
        0.0, -1.0,
        1.0, 0.0
    );

    float2 cross1 = dot(mul(m, c - b), p - b);
    float2 cross2 = dot(mul(m, c - b), a - b);

    return uint((dot(cross1, cross2) >= 0));

}

public uint inTriangleTest(float2 p, float2 a, float2 b, float2 c)
{
    return uint(side_test(p, a, b, c) & side_test(p, b, a, c) & side_test(p, c, a, b));
}
/**
 * Rasterize a single triangle.
 * @param tid The 2D coordinate of the current thread. Ranges from (0, 0) to (canvasSize.x - 1, canvasSize.y - 1).
 * @param canvasSize The size of the output image in pixels.
 * @param v0 The first vertex of the triangle in normalized coordinates.
 * @param v1 The second vertex of the triangle in normalized coordinates.
 * @param v2 The third vertex of the triangle in normalized coordinates.
 * @return The color white if the pixel is inside the triangle, otherwise black.
 */
public float4 triangleRasterizer(uint2 tid, uint2 canvasSize, float2 v0, float2 v1, float2 v2)
{
    float2 tid = float2(tid) / float2(canvasSize);

    float3 result = float3(1.0, 1.0, 1.0);

    result = inTriangleTest(tid, v0, v1, v2) * result;

    return float4(result, 1.0);

    // TODO: Student implementation ends here.
}

/**
 * Test ray-triangle intersection for a single triangle.
 * @param tid The 2D coordinate of the current thread. Ranges from (0, 0) to (canvasSize.x - 1, canvasSize.y - 1).
 * @param canvasSize The size of the output image in pixels.
 * @param triangles A buffer containing triangles to test against. Only the first triangle is used in this function.
 * @return The color of the triangle if hit, otherwise black.
 */
public float4 rayTriangleTest(uint2 tid, uint2 canvasSize, StructuredBuffer<Triangle> triangles)
{
    // Get the first triangle in the buffer.
    Triangle triangle = triangles[0];
    // We generate rays with origin from (0, 0, 0) to (1, 1, 0) and direction towards the negative z axis (0, 0, -1).
    float2 uv = float2(tid) / float2(canvasSize);
    Ray ray = Ray(float3(uv, 0.0), float3(0.0, 0.0, -1.0), float2(0.0, float.maxValue));
    // How can we make sure the rays are set up correctly? Just return the value as color.
    // return float4(ray.origin.xy, 0.0, 1.0);
    Optional<RayHitResult> hitResult = triangle.hit(ray);
    if (hitResult.hasValue)
    {
        return float4(triangle.averageColor(), 1.0);
    }
    else
    {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
}

/**
 * Test ray-triangle intersection for multiple triangles.
 * @param tid The 2D coordinate of the current thread. Ranges from (0, 0) to (canvasSize.x - 1, canvasSize.y - 1).
 * @param canvasSize The size of the output image in pixels.
 * @param triangles A buffer containing triangles to test against.
 * @param triangleCount The number of triangles in the buffer.
 * @return The color of the closest triangle if hit, otherwise black.
 */
public float4 rayTrianglesTest(uint2 tid, uint2 canvasSize, StructuredBuffer<Triangle> triangles, uint triangleCount)
{
    // We generate rays with origin from (0, 0, 0) to (1, 1, 0) and direction towards the negative z axis (0, 0, -1).
    float2 uv = float2(tid) / float2(canvasSize);
    Ray ray = Ray(float3(uv, 0.0), float3(0.0, 0.0, -1.0), float2(0.0, float.maxValue));

    // find the normal of the plane defined by the triangle
    Triangle closest;
    float t_smallest = float.maxValue;
    for (int i = 0; i < triangleCount; i++)
    {
        Optional<RayHitResult> hitResult = triangles[i].hit(ray);
        if (hitResult.hasValue) {
            if (hitResult.value.t < t_smallest)
            {
                t_smallest = hitResult.value.t;
                closest = triangles[i];

            }
        }
    }

    if (t_smallest == float.maxValue)
    {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    return float4(closest.averageColor(), 1.0);

    // TODO: Student implementation starts here.
}
