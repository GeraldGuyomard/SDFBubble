/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
The app's source code for its Metal shaders.
*/

#import <metal_stdlib>

#import "ShaderTypes.h"

using namespace metal;

/// A type that stores the vertex shader's output and serves as an input to the fragment shader.
struct RasterizerData
{
    /// A 4D position in clip space from a vertex shader function.
    ///
    /// The `[[position]]` attribute indicates that the position is the vertex's
    /// clip-space position.
    float4 position [[position]];

    /// A texture coordinate value, either for a vertex as an output from a vertex shader,
    /// or for a fragment as input to a fragment shader.
    ///
    /// As an input to a fragment shader, the rasterizer interpolates the coordinate
    /// values between the triangle's vertices for each fragment because this
    /// member doesn't have a special attribute.
    float2 textureCoordinate;

};

/// Converts each input vertex from pixel coordinates to clip-space coordinates.
///
/// The vertex shader doesn't modify the texture coordinate values.
vertex RasterizerData
vertexShader(uint                   vertexID              [[ vertex_id ]],
             constant VertexData    *vertexArray          [[ buffer(BufferBindingIndexForVertexData) ]],
             constant Uniforms    *uniforms  [[ buffer(BufferBindingIndexForUniforms) ]])

{
    /// The vertex shader's return value.
    RasterizerData out;

    // Retrieve the 2D position of the vertex in pixel coordinates.
    simd_float2 pixelSpacePosition = vertexArray[vertexID].position.xy;

    // Retrieve the viewport's size by casting it to a 2D float value.
    simd_float2 viewportSize = uniforms->viewportSize;

    // Convert the position in pixel coordinates to clip-space by dividing the
    // pixel's coordinates by half the size of the viewport.
    out.position.xy = pixelSpacePosition / (viewportSize / 2.0);
    out.position.z = 0.0;
    out.position.w = 1.0;

    // Pass the input texture coordinates directly to the rasterizer.
    out.textureCoordinate = vertexArray[vertexID].textureCoordinate;

    return out;
}

/// A Returns color data from the input texture by sampling it at the fragment's
/// texture coordinates.
fragment float4 samplingShader(RasterizerData  in           [[stage_in]],
                               texture2d<half> colorTexture [[ texture(RenderTextureBindingIndex) ]])
{
    /// A basic texture sampler with linear filter settings.
    constexpr sampler textureSampler (mag_filter::linear,
                                      min_filter::linear);

    /// The color value of the input texture at the fragment's texture coordinates.
    const half4 colorSample = colorTexture.sample (textureSampler, in.textureCoordinate);

    // Pass the texture color to the rasterizer.
    return (simd_float4)(colorSample);
}

float opUnion( float d1, float d2 )
{
    return min(d1,d2);
}

float opSmoothUnion( float d1, float d2, float k )
{
    k *= 4.0;
    float h = max(k-abs(d1-d2),0.0);
    return min(d1, d2) - h*h*0.25/k;
}

float computeSDF(Bubble bubble1, Bubble bubble2, float smoothFactor, float2 pt)
{
    return opSmoothUnion(bubble1.computeSDF(pt), bubble2.computeSDF(pt), smoothFactor);
    //return opUnion(bubble1.computeSDF(pt), bubble2.computeSDF(pt));
}

bool evaluateBubbleGroup(constant BubbleGroup& group,
                    constant Bubble* bubbles,
                    float2 pt,
                    uint2 gridId,
                    texture2d<half, access::read_write> texture)
{
    float d;
    if (group.nbBubbles == 1)
    {
        const auto b = bubbles[0];
        d = b.computeSDF(pt);
    }
    else if (group.nbBubbles == 2)
    {
        const auto b1 = bubbles[0];
        const auto b2 = bubbles[1];
        d = computeSDF(b1, b2, group.smoothFactor, pt);
    }
    else
    {
        // blend
        return false;
    }
    
    
    if (d <= 0.f)
    {
        half4 c = texture.read(gridId);
        
        // inside
        c += half4 { 0.1f, 0.1f, 0.1f, 0.f};

        texture.write(c, gridId);
        return true;
    }
    
    return false;
}

kernel void
computeAndDrawSDF(texture2d<half, access::read_write> texture [[texture(ComputeTextureBindingIndexForColorImage)]],
                uint2 gridId     [[thread_position_in_grid]],
               constant Uniforms* uniforms  [[ buffer(BufferBindingIndexForUniforms) ]])
{

    // Check that that this part of the grid is within the texture's bounds.
    if ((gridId.x >= texture.get_width()) ||
        (gridId.y >= texture.get_height()))
    {
        // Exit early for coordinates outside the bounds of the destination.
        return;
    }

    const float2 pt { float(gridId.x), float(gridId.y) };
    constant Bubble* bubbles = &uniforms->bubbles[0];
    
    for (size_t i=0; i < uniforms->nbBubbleGroups; ++i)
    {
        constant auto& group = uniforms->groups[i];
        if (evaluateBubbleGroup(group, bubbles, pt, gridId, texture))
        {
            break;
        }
        
        bubbles += group.nbBubbles;
    }
}
