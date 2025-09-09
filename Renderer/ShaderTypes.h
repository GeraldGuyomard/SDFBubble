/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
The types and enum constants the app shares with its Metal shaders and C/ObjC code.
*/

#ifndef ShaderTypes_h
#define ShaderTypes_h

#import <simd/simd.h>

#if defined(__METAL_VERSION__)
    #define SHADER_CONSTANT constant
    using namespace metal;
#else
    #define SHADER_CONSTANT const
    using namespace simd;
#endif

/// Defines the binding index values for passing buffer arguments to GPU function parameters.
///
/// The binding values define an agreement between:
/// - The app's main code in Objective-C that submits the data to the GPU
/// - The shader code that defines the GPU functions, which receive the data through their parameters
///
/// The values need to match between both sides of the exchange for the data to get
/// to the correct place.
enum BufferBindingIndex
{
    /// The buffer binding index value that stores the triangle's vertex data.
    ///
    /// The data at this binding index stores an array of three ``VertexData`` instances.
    BufferBindingIndexForVertexData = 0,

    /// The buffer binding index value that stores the app's viewport's size.
    ///
    /// The vertex shader calculates the pixel coordinates of the triangle's vertices
    /// based on the size of the app's viewport.
    BufferBindingIndexForUniforms = 1,
};

/// Defines the binding index values for passing texture arguments to GPU function parameters.
///
/// The binding values define an agreement between:
/// - The app's main code in Objective-C that submits the data to the GPU
/// - The shader code that defines the GPU functions, which receive the data through their parameters
///
/// The values need to match between both sides of the exchange for the data to get
/// to the correct place.
enum TextureBindingIndex
{
    /// An index of a color texture for a compute kernel in a compute pass.
    ComputeTextureBindingIndexForColorImage = 0,

    /// An index of a texture for a fragment shader in a render pass.
    RenderTextureBindingIndex = 0,
};

/// A type that defines the data layout for a triangle vertex,
/// which includes position and texture coordinate values.
///
/// The app's main code and shader code apply this type for data layout consistency.
struct VertexData
{
    /// The location for a vertex in 2D, pixel-coordinate space.
    ///
    /// For example, a value of `100` in either dimension means the vertex is
    /// 100 pixels from the origin in that dimension.
    float2 position;

    /// The location within a 2D texture for a vertex.
    float2 textureCoordinate;
};

struct Bubble final
{
    float2 origin;
    float radius;
    
    float computeSDF(float2 pt) SHADER_CONSTANT
    {
        const float d = length(pt - origin) - radius;
        return d;
    }
};

struct BubbleGroup final
{
    size_t nbBubbles = 0;
    float smoothFactor = 50.f;
};

struct Uniforms final
{
    float2 viewportSize;
    size_t nbBubbleGroups;
    
    BubbleGroup groups[1024];
    Bubble bubbles[1024];
};

float opUnion( float d1, float d2 )
{
    return min(d1,d2);
}

float opSmoothUnion( float d1, float d2, float k )
{
    k *= 4.0;
    float h = max(k-abs(d1-d2),0.0f);
    return min(d1, d2) - h*h*0.25f/k;
}

float computeSDF(SHADER_CONSTANT Bubble& bubble1, SHADER_CONSTANT Bubble& bubble2, float smoothFactor, float2 pt)
{
    return opSmoothUnion(bubble1.computeSDF(pt), bubble2.computeSDF(pt), smoothFactor);
    //return opUnion(bubble1.computeSDF(pt), bubble2.computeSDF(pt));
}

float computeSDF(SHADER_CONSTANT Bubble* bubble, size_t nbBubbles, float smoothFactor, float2 pt)
{
#if 1
    float d = bubble[0].computeSDF(pt);
    
    for (size_t i = 1; i < nbBubbles; ++i)
    {
        const float newD = bubble[i].computeSDF(pt);
        //d = opSmoothUnion(d, (bubble++)->computeSDF(pt), smoothFactor);
        d = opUnion(d, newD);
    }
    
#else
    SHADER_CONSTANT Bubble* const end = bubble + nbBubbles;
    
    float d = (bubble++)->computeSDF(pt);
    
    while (bubble < end)
    {
        //d = opSmoothUnion(d, (bubble++)->computeSDF(pt), smoothFactor);
        d = opUnion(d, (bubble++)->computeSDF(pt));
    }
#endif
    
    return d;
}

template <typename TTextureAccessor>
bool evaluateBubbleGroup(SHADER_CONSTANT BubbleGroup& group,
                    SHADER_CONSTANT Bubble* bubbles,
                    float2 pt,
                    TTextureAccessor accessor)
{
    float d;
    if (group.nbBubbles == 1)
    {
        d = bubbles[0].computeSDF(pt);
    }
    else if (group.nbBubbles == 2)
    {
        d = computeSDF(bubbles[0], bubbles[1], group.smoothFactor, pt);
    }
    else
    {
        // blend
        return computeSDF(&bubbles[0], group.nbBubbles, group.smoothFactor, pt);
    }
    
    
    if (d <= 0.f)
    {
        half4 c = accessor.read();
        
        // inside
        c += half4 { 0.1f, 0.1f, 0.1f, 0.f};

        accessor.write(c);
        return true;
    }
    
    return false;
}

#endif /* ShaderTypes_h */
