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

float computeSDFStrength(float sdf, float k, float m)
{
    return (sdf != 0.f) ? m * exp(sdf * k) : 0.f;
}

/// A Returns color data from the input texture by sampling it at the fragment's
/// texture coordinates.
fragment float4 samplingShader(RasterizerData  in           [[stage_in]],
                               texture2d<half> colorTexture [[ texture(RenderTextureBindingIndex) ]],
                               texture2d<half> sdfGradientTexture [[ texture(SDFGradientTextureBindingIndex) ]],
                               constant Uniforms* uniforms  [[ buffer(BufferBindingIndexForUniforms) ]])
{
    /// A basic texture sampler with linear filter settings.
    constexpr sampler textureSampler (mag_filter::linear,
                                      min_filter::linear);

    const half4 colorSample = colorTexture.sample (textureSampler, in.textureCoordinate);
    
    const half4 distanceAndGradient = sdfGradientTexture.sample (textureSampler, in.textureCoordinate);
    
    auto c = colorSample;
    
    const float sdf = distanceAndGradient.x;
    
    // refractions
    const float2 gradient { distanceAndGradient.y, distanceAndGradient.z };
    const half gradientStrength = computeSDFStrength(sdf, 1e-2f, 0.8f);
    
    if (gradientStrength >= 0.025f)
    {
        const float2 v = gradient * uniforms->gradientScale * 1e1f;
        const auto refractionPos = in.textureCoordinate + v;
        const auto refractionColor = colorTexture.sample(textureSampler, refractionPos);
        
        c = refractionColor;
        //c = half4 { 1.f, 0.f, 0.f, 1.f };
        //c = mix(colorSample, refractionColor, gradientStrength);
    }
    
    const half luminosity = computeSDFStrength(sdf, 1e-1f, 0.8f);
    c += half4 { luminosity, luminosity, luminosity, 0.f };
    
    // Pass the texture color to the rasterizer.
    return (float4) c;
}

template <typename TPixel, metal::access TAccess>
class MetalTextureAccessor final
{
public:
    using TTexture = texture2d<TPixel, TAccess>;
    
    MetalTextureAccessor(TTexture texture, uint2 gridId)
    : _texture(texture), _gridId(gridId)
    {}
    
    TPixel read(uint2 pos)
    {
        return _texture.read(pos).r;
    }
    
    void write(TPixel v) const
    {
        _texture.write(v, _gridId);
    }
    
    void writeFloat4(float4 v) const
    {
        _texture.write(v, _gridId);
    }
    
    bool isValid() const
    {
        return (_gridId.x < _texture.get_width()) && (_gridId.y < _texture.get_height());
    }
    
    float2 position() const
    {
        return { float(_gridId.x), float(_gridId.y) };
    }
    
    uint2 gridId() const
    {
        return _gridId;
    }
    
private:
    TTexture _texture;
    uint2 _gridId;
};

kernel void
computeAndDrawSDF(texture2d<half, access::write> texture [[texture(ComputeTextureBindingIndexForSDF)]],
                uint2 gridId     [[thread_position_in_grid]],
               constant Uniforms* uniforms  [[ buffer(BufferBindingIndexForUniforms) ]])
{

    MetalTextureAccessor accessor { texture, gridId };
    
    computeAndDrawSDF(accessor, uniforms);
}

kernel void drawSDFGradient(texture2d<half, access::read> sdfTextureIn [[texture(ComputeTextureBindingIndexForSDF)]],
                            texture2d<float, access::write> sdfGradientTextureOut [[texture(ComputeTextureBindingIndexForGradientSDF)]],
                            uint2 gridId     [[thread_position_in_grid]])
{
    MetalTextureAccessor accessorIn { sdfTextureIn, gridId };
    MetalTextureAccessor accessorOut { sdfGradientTextureOut, gridId };
    
    drawSDFGradient(accessorIn, accessorOut);
}
