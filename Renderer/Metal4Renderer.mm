#import <simd/simd.h>
#import <MetalKit/MetalKit.h>

#import <Metal/MTL4RenderPass.h>
#import "UIKit/UIKit.h"
#import <CoreMotion/CoreMotion.h>

#import "Metal4Renderer.h"
#import "TGAImage.h"

#include <algorithm>
#include <vector>
#include <set>
#include <optional>

#import "ShaderTypes.h"

using namespace simd;

static const MTLOrigin zeroOrigin = { 0, 0, 0 };

constexpr uint32_t kMaxFramesInFlight = 3;

class BubbleSet final
{
public:
    
    void add(const float2& origin, float radius)
    {
        Bubble b { origin, radius };
        b.id = (uint32_t) _bubbles.size();
        
        _bubbles.push_back(b);
    }
    
    void remove(Bubble& bubble)
    {
        const auto end = _bubbles.end();
        for (auto it = _bubbles.begin(); it != end; ++it)
        {
            auto& b = *it;
            if (&b == &bubble)
            {
                _bubbles.erase(it);
                break;
            }
        }
    }
    
    Bubble* pick(const float2& pos)
    {
        const size_t n = _bubbles.size();
        for (size_t i=0; i < n; ++i)
        {
            Bubble& bubble = _bubbles[i];
            const float d = bubble.computeSDF(pos);
            if (d <= 0.f)
            {
                return &bubble;
            }
        }
        
        return nullptr;
    }
    
    void setSelection(Bubble& bubble, const float2& initialHitInSDFSpace)
    {
        _selection = Selection { bubble, initialHitInSDFSpace };
    }
    
    void clearSelection()
    {
        _selection.reset();
    }
    
    void moveSelection(const float2& pt)
    {
        if (_selection.has_value())
        {
            const auto delta = pt - _selection->initialHitInSDFSpace;
            _selection->bubble->origin = _selection->initialOrigin + delta;
        }
    }
    
    void rescaleSelection(float scale)
    {
        if (_selection.has_value())
        {
            _selection->bubble->radius = _selection->initialRadius * scale;
        }
    }
    
    void update(Uniforms& uniforms)
    {
        std::vector<BubbleGroup> groups;
        
        // brute force N square comparison
        std::set<const Bubble*> bubblesSet;
        for (const auto& b : _bubbles)
        {
            bubblesSet.insert(&b);
        }
        
        std::vector<Bubble> bubbles;
        
        while (!bubblesSet.empty())
        {
            auto it = bubblesSet.begin();
            
            const auto* bubble = *it;
            bubblesSet.erase(it);
            
            const size_t startIndex = bubbles.size();
            bubbles.push_back(*bubble);
            
            BubbleGroup group;
            group.nbBubbles = 1;
            
            float minD = std::numeric_limits<float>::max();
            
            // find all other bubbles interacting with the main bubble
            for (auto it = bubblesSet.begin(); it != bubblesSet.end();)
            {
                const auto* otherBubble = *it;
                bool coalesced = false;
                
                for (size_t i=0; i < group.nbBubbles; ++i)
                {
                    const auto& b = bubbles[startIndex + i];
                    const float distance = length(b.origin - otherBubble->origin);
                    
                    const float minDist = b.radius + otherBubble->radius;
                    
                    minD = std::min(minD, distance);
                    
                    if (distance <= minDist)
                    {
                        ++group.nbBubbles;
                        bubbles.push_back(*otherBubble);
                        it = bubblesSet.erase(it);
                        coalesced = true;
                        break;
                    }
                }
                
                if (!coalesced)
                {
                    ++it;
                }
            }
            
            if (group.nbBubbles > 1)
            {
                group.smoothFactor = 3e3f / (1.f + minD);
            }
            
            groups.push_back(group);
        }
        
        const size_t nbGroups = groups.size();
        uniforms.nbBubbleGroups = nbGroups;
        
        for (size_t i=0; i < nbGroups; ++i)
        {
            uniforms.groups[i] = groups[i];
        }
        
        const size_t nbBubbles = bubbles.size();
        for (size_t i=0; i < nbBubbles; ++i)
        {
            uniforms.bubbles[i] = bubbles[i];
        }
    }
    
private:
    std::vector<Bubble> _bubbles;
    
    struct Selection final
    {
        Selection(Bubble& bubble, const float2& initialHitInSDFSpace)
        : bubble(&bubble),
        initialOrigin(bubble.origin),
        initialRadius(bubble.radius),
        initialHitInSDFSpace(initialHitInSDFSpace)
        {}
        
        Bubble* bubble;
        float2 initialOrigin;
        float initialRadius;
        
        float2 initialHitInSDFSpace;
    };
    
    std::optional<Selection> _selection;
};

@interface Metal4Renderer()
@end

/// A class that renders each of the app's video frames.
@implementation Metal4Renderer
{
    UIView* view;
    
    /// A Metal device the renderer draws with by sending commands to it.
    id<MTLDevice> device;
    
    /// A Metal compiler that compiles the app's shaders into pipelines.
    id<MTL4Compiler> compiler;

    /// A default library that stores the app's shaders and compute kernels.
    ///
    /// Xcode compiles the shaders from the project's `.metal` files at build time
    /// and stores them in the default library inside the app's main bundle.
    id<MTLLibrary> defaultLibrary;

    /// A compute pipeline the app creates at runtime.
    ///
    /// The app compiles the pipeline with the compute kernel in the
    /// `AAPLShaders.metal` source code file.
    id<MTLComputePipelineState> drawSDFPipelineState;
    id<MTLComputePipelineState> drawSDFGradientPipelineState;
    
    /// A render pipeline the app creates at runtime.
    ///
    /// The app compiles the pipeline with the vertex and fragment shaders in the
    /// `AAPLShaders.metal` source code file.
    id<MTLRenderPipelineState> renderPipelineState;

    /// A command queue the app uses to send command buffers to the Metal device.
    id<MTL4CommandQueue> commandQueue;

    /// An array of allocators, each of which manages memory for a command buffer.
    ///
    /// Each allocator provides backing memory for the commands the app encodes
    /// into a command buffer..
    id<MTL4CommandAllocator> commandAllocators[kMaxFramesInFlight];

    /// A reusable command buffer the render encodes draw commands to for each frame.
    id<MTL4CommandBuffer> commandBuffer;

    /// An argument table that stores the resource bindings for both
    /// render and compute encoders.
    id<MTL4ArgumentTable> argumentTable;

    /// A residency set that keeps resources in memory for the app's lifetime.
    id<MTLResidencySet> residencySet;

    /// A shared event that the GPU signals to indicate to the CPU that it's
    /// finished work.
    id<MTLSharedEvent> sharedEvent;

    /// An integer that tracks the current frame number.
    uint64_t frameNumber;
    
    /// A texture that stores the original background image.
    ///
    /// The app build a color image by combines this texture with
    /// `chyronTexture`, which becomes the input texture for the grayscale conversion.
    id<MTLTexture> backgroundImageTexture;
    id<MTLTexture> sdfTexture;
    id<MTLTexture> sdfGradientTexture;

    /// A two-dimensional size that represents the number of threads for each
    /// grid dimension of a threadgroup for a compute kernel dispatch.
    MTLSize threadgroupSize;

    /// A two-dimensional size that represents the number of threads in a
    /// threadgroup for a compute kernel dispatch.
    MTLSize threadgroupCount;

    /// A buffer that stores the triangle vertex data for the render pass.
    ///
    /// The app stores a copy of the data from `triangleVertexData`.
    id<MTLBuffer> vertexDataBuffer;

    /// The current size of the view, which the app sends as an input to the
    /// vertex shader.
    float2 viewportSize;

    /// A buffer that stores the viewport's size data.
    ///
    id<MTLBuffer> uniformsBuffer;
    
    BubbleSet _bubbleSet;
    
    UIPanGestureRecognizer* panGestureRecognizer;
    UITapGestureRecognizer* tapGestureRecognizer;
    UITapGestureRecognizer* doubleTapGestureRecognizer;
    
    CMMotionManager* motionManager;
    float2 lightDirection;
}

/// Creates a texture instance from an image file.
///
/// The method configures the texture with a pixel format with 4 color channels:
/// - blue
/// - green
/// - red
/// - alpha
///
/// Each channel is an 8-bit unnormalized value.
///
/// For example:
/// - `0` maps to `0.0`,
/// - `255` maps to `1.0`.
///
/// - Returns: A texture instance if the method succeeds; otherwise `nil`.
- (id<MTLTexture>)loadImageToTexture:(NSURL *)imageFileLocation
{
    // Load an image from a URL.
    TGAImage *image;
    image = [[TGAImage alloc] initWithTGAFileAtLocation:imageFileLocation];

    if (!image)
    {
        return nil;
    }

    // Create and configure the texture descriptor to make a texture that's the
    // same size as the image.
    MTLTextureDescriptor *textureDescriptor = [[MTLTextureDescriptor alloc] init];

    textureDescriptor.pixelFormat = MTLPixelFormatBGRA8Unorm;
    textureDescriptor.textureType = MTLTextureType2D;
    textureDescriptor.usage = MTLTextureUsageShaderRead;
    textureDescriptor.width = image.width;
    textureDescriptor.height = image.height;

    // Create the texture instance.
    id<MTLTexture> texture;
    texture = [device newTextureWithDescriptor:textureDescriptor];

    if (nil == texture)
    {
        NSLog(@"The device can't create a texture for the image at: %@",
              imageFileLocation);
        return nil;
    }

    // Define a region that's the size of the texture, which is the same as the image.
    const MTLSize size = {
        textureDescriptor.width,
        textureDescriptor.height,
        1
    };
    const MTLRegion region = { zeroOrigin, size };

    /// The number of bytes in each of the texture's rows.
    NSUInteger bytesPerRow = 4 * textureDescriptor.width;

    // Copy the bytes from the image into the texture.
    [texture replaceRegion:region
               mipmapLevel:0
                 withBytes:image.data.bytes
               bytesPerRow:bytesPerRow];

    return texture;
}

/// Creates a compiler to create pipelines from shaders.
- (void) createCompiler
{
    MTL4CompilerDescriptor *compilerDescriptor;
    compilerDescriptor = [[MTL4CompilerDescriptor alloc] init];
    
    // Create a compiler with the descriptor.
    NSError *error = NULL;
    compiler = [device newCompilerWithDescriptor:compilerDescriptor
                                            error:&error];
    
    // Verify the device created the compiler successfully.
    NSAssert(nil != compiler,
             @"The device can't create a compiler due to: %@",
             error);
}

- (id<MTLComputePipelineState>)createComputePipelineStateWithFunctionName:(NSString*)name
{
    NSError *error = NULL;

    // Get the kernel function from the default library.
    MTL4LibraryFunctionDescriptor *kernelFunction;
    kernelFunction = [MTL4LibraryFunctionDescriptor new];
    kernelFunction.library = defaultLibrary;
    kernelFunction.name = name;
    
    // Configure a compute pipeline with the compute function.
    MTL4ComputePipelineDescriptor *pipelineDescriptor;
    pipelineDescriptor = [MTL4ComputePipelineDescriptor new];
    pipelineDescriptor.computeFunctionDescriptor = kernelFunction;

    // Create a compute pipeline with the image processing kernel in the library.
    id<MTLComputePipelineState> state = [compiler newComputePipelineStateWithDescriptor:pipelineDescriptor
                                                       compilerTaskOptions:nil
                                                                     error:&error];
    
    // Verify the compiler created the pipeline state successfully.
    // Debug builds in Xcode turn on Metal API Validation by default.
    NSAssert(nil != state,
             @"The compiler can't create a compute pipeline with kernel function: %@",
             name);
    
    return state;
}

/// Creates a compute pipeline with a kernel function.

- (void) createRenderPipelineFor:(MTLPixelFormat)pixelFormat
{
    NSError *error = NULL;

    // Get the vertex function from the default library.
    MTL4LibraryFunctionDescriptor *vertexFunction;
    vertexFunction = [MTL4LibraryFunctionDescriptor new];
    vertexFunction.library = defaultLibrary;
    vertexFunction.name = @"vertexShader";

    // Get the fragment function from the default library.
    MTL4LibraryFunctionDescriptor *fragmentFunction;
    fragmentFunction = [MTL4LibraryFunctionDescriptor new];
    fragmentFunction.library = defaultLibrary;
    fragmentFunction.name = @"samplingShader";

    // Configure a render pipeline with the vertex and fragment shaders.
    MTL4RenderPipelineDescriptor *pipelineDescriptor;
    pipelineDescriptor = [MTL4RenderPipelineDescriptor new];
    pipelineDescriptor.label = @"Simple Render Pipeline";
    pipelineDescriptor.vertexFunctionDescriptor = vertexFunction;
    pipelineDescriptor.fragmentFunctionDescriptor = fragmentFunction;
    pipelineDescriptor.colorAttachments[0].pixelFormat = pixelFormat;

    renderPipelineState = [compiler newRenderPipelineStateWithDescriptor:pipelineDescriptor
                                                     compilerTaskOptions:nil
                                                                   error:&error];

    NSAssert(nil != renderPipelineState,
             @"The compiler can't create a render pipeline due to: %@",
             error);
}

- (void) createBuffers
{
    const float2 contentSize { float(backgroundImageTexture.width), float(backgroundImageTexture.height) };
    const float2 vSize { float(viewportSize.x), float(viewportSize.y) };
    
    const float2 s = vSize / contentSize;
    const float minS = std::min(s.x, s.y);
    float rescaledContentSizeX = contentSize.x * minS;
    const float offsetX = (vSize.x - rescaledContentSizeX) * 0.5f;
    
    const float w = (vSize.x - offsetX) * 0.5f;
    const float h = w * contentSize.y / contentSize.x;
    
    const VertexData triangleVertexData[] =
    {
        { {  w,  -h },  { 1.f, 1.f } },
        { { -w,  -h },  { 0.f, 1.f } },
        { { -w,  h },  { 0.f, 0.f } },

        // The 2nd triangle of the rectangle for the composite color texture.
        { {  w,  -h },  { 1.f, 1.f } },
        { { -w,  h },  { 0.f, 0.f } },
        { {  w,  h },  { 1.f, 0.f } },

    };
    
    // Create the buffer that stores the vertex data.
    vertexDataBuffer = [device newBufferWithLength:sizeof(triangleVertexData)
                                             options:MTLResourceStorageModeShared];

    memcpy(vertexDataBuffer.contents, triangleVertexData, sizeof(triangleVertexData));

    // Create the buffer that stores the app's viewport data.
    uniformsBuffer = [device newBufferWithLength:sizeof(Uniforms) options:MTLResourceStorageModeShared];

    [self updateUniformsBuffer];
}

/// Loads two textures the app combines into the source color texture.
- (void) createTextures
{
    //NSString *backgroundImageFileName = @"Hawaii-coastline";
    NSString *backgroundImageFileName = @"water";
    // Create a texture from the background image file.
    NSURL *backgroundImageFile = [[NSBundle mainBundle]
                                URLForResource:backgroundImageFileName
                                withExtension:@"tga"];
    backgroundImageTexture = [self loadImageToTexture:backgroundImageFile];
    NSAssert(nil != backgroundImageTexture,
             @"The app can't create a texture for the background image: %@",
             backgroundImageFileName);
    backgroundImageTexture.label = @"BackgroundImageTexture";
    
    // Create the source color texture that stores the combined texture data.
    MTLTextureDescriptor *textureDescriptor = [[MTLTextureDescriptor alloc] init];
    textureDescriptor.textureType = MTLTextureType2D;

    // Configure the pixel format with 4 channels: blue, green, red, and alpha.
    // Each is an 8-bit, unnormalized value; `0` maps to `0.0` and `255` maps to `1.0`.
    textureDescriptor.pixelFormat = MTLPixelFormatBGRA8Unorm;
    textureDescriptor.width = backgroundImageTexture.width;
    textureDescriptor.height = backgroundImageTexture.height;

    // Configure the input texture to read-only because `convertToGrayscale` kernel
    // doesn't modify it.
    textureDescriptor.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    
    textureDescriptor.pixelFormat = MTLPixelFormatR16Float;
    sdfTexture = [device newTextureWithDescriptor:textureDescriptor];
    NSAssert(nil != sdfTexture,
             @"The device can't create a texture for the SDF.");
    sdfTexture.label = @"SDF Texture";
    
    textureDescriptor.pixelFormat = MTLPixelFormatRGBA16Float;
    sdfGradientTexture = [device newTextureWithDescriptor:textureDescriptor];
    NSAssert(nil != sdfGradientTexture,
             @"The device can't create a texture for the gradient.");
    sdfGradientTexture.label = @"SDF Gradient Texture";
    
    
}

/// Configures the number of rows and columns in the threadgroups based on the input image's size.
///
/// The method ensures the grid covers an area that's at least as big as the
/// entire image.
- (void) configureThreadgroupForComputePasses
{
    NSAssert(backgroundImageTexture, @"Create the composite color texture before configuring the threadgroup");

    // Set the compute kernel's threadgroup size to 16 x 16.
    threadgroupSize = MTLSizeMake(16, 16, 1);

    // Find the number of threadgroup widths the app needs to span the texture's full width.
    threadgroupCount.width  = backgroundImageTexture.width  + threadgroupSize.width -  1;
    threadgroupCount.width /= threadgroupSize.width;

    // Find the number of threadgroup heights the app needs to span the texture's full width.
    threadgroupCount.height = backgroundImageTexture.height + threadgroupSize.height - 1;
    threadgroupCount.height /= threadgroupSize.height;

    // Set depth to one because the image data is two-dimensional.
    threadgroupCount.depth = 1;
}

- (void) createArgumentTable
{
    // Create an argument table that stores 2 buffers and 2 textures.
    MTL4ArgumentTableDescriptor *argumentTableDescriptor;
    argumentTableDescriptor = [[MTL4ArgumentTableDescriptor alloc] init];

    // Configure the descriptor to store 2 buffers:
    // - A vertex buffer
    // - A viewport size buffer.
    argumentTableDescriptor.maxTextureBindCount = 4;
    argumentTableDescriptor.maxBufferBindCount = 2;

    // Create an argument table with the descriptor.
    NSError *error = NULL;
    argumentTable = [device newArgumentTableWithDescriptor:argumentTableDescriptor
                                                     error:&error];
    NSAssert(nil != argumentTable,
             @"The device can't create an argument table due to: %@", error);
}

- (void)createSharedEvent
{
    // Initialize the shared event to permit the renderer to start on the first frame.
    sharedEvent = [device newSharedEvent];
    sharedEvent.signaledValue = frameNumber;
}


- (void) createResidencySets
{
    NSError *error = NULL;

    // Create a communal residency set for resources the app needs for every frame.
    MTLResidencySetDescriptor *residencySetDescriptor;
    residencySetDescriptor = [MTLResidencySetDescriptor new];
    residencySet = [device newResidencySetWithDescriptor:residencySetDescriptor
                                                     error:&error];

    NSAssert(nil != residencySet,
             @"The device can't create a residency set due to: %@", error);

    // Add the communal residency set to the command queue.
    [commandQueue addResidencySet:residencySet];

    // Add the communal resources to the residency set.
    [residencySet addAllocation:backgroundImageTexture];
    [residencySet addAllocation:sdfTexture];
    [residencySet addAllocation:sdfGradientTexture];
    [residencySet addAllocation:vertexDataBuffer];
    [residencySet addAllocation:uniformsBuffer];
    [residencySet commit];
    
    // Create per-frame allocators and residency sets.
    for (uint32_t i = 0; i < kMaxFramesInFlight; i++)
    {
        commandAllocators[i] = [device newCommandAllocator];
        NSAssert(nil != commandAllocators[i],
                 @"The device can't create an allocator set due to: %@", error);
    }
}

- (nonnull instancetype)initWithView:(nonnull MTKView *)mtkView
{
    self = [super init];
    if (nil == self) { return nil; }

    frameNumber = 0;
    viewportSize.x = (simd_uint1)mtkView.drawableSize.width;
    viewportSize.y = (simd_uint1)mtkView.drawableSize.height;
    
    view = mtkView;
    device = mtkView.device;
    
    commandQueue = [device newMTL4CommandQueue];
    commandBuffer = [device newCommandBuffer];
    defaultLibrary = [device newDefaultLibrary];

    // Create the app's resources.
    [self createTextures];
    
    /*
    const float2 size { float(offscreenTexture.width), float(offscreenTexture.height) };
    
    _bubbles.push_back({
        .origin = size * 0.5f,
        .radius = 200.f
    });
    
    _bubbles.push_back({
        .origin = size * 0.75f,
        .radius = 100.f
    });
    
    _bubbles.push_back({
        .origin = size * 0.25f,
        .radius = 150.f
    });
    
    Bubble b1 {
        .origin = float2 { size.x * 0.7f, size.y * 0.2f },
        .radius = 100.f
    };
    
    _bubbles.push_back(b1);
    
    _bubbles.push_back( {
        .origin = b1.origin + float2{ 120.f, 120.f },
        .radius = 80.f
    });*/
    
    [self createBuffers];

    // Create the types that manage the resources.
    [self createArgumentTable];
    [self createSharedEvent];
    [self createResidencySets];
    
    // Add the Metal layer's residency set to the queue.
    [commandQueue addResidencySet:((CAMetalLayer *)mtkView.layer).residencySet];

    // Create the compute pipeline.
    [self createCompiler];
    
    drawSDFPipelineState = [self createComputePipelineStateWithFunctionName:@"computeAndDrawSDF"];
    drawSDFGradientPipelineState = [self createComputePipelineStateWithFunctionName:@"drawSDFGradient"];
    
    [self configureThreadgroupForComputePasses];

    // Configure the view's color format.
    const MTLPixelFormat pixelFormat = MTLPixelFormatBGRA8Unorm_sRGB;
    mtkView.colorPixelFormat = pixelFormat;

    // Create the render pipeline.
    [self createRenderPipelineFor:pixelFormat];
    
    panGestureRecognizer = [[UIPanGestureRecognizer alloc] initWithTarget:self action:@selector(onPan:)];
    [mtkView addGestureRecognizer:panGestureRecognizer];
    
    tapGestureRecognizer = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(onTap:)];
    tapGestureRecognizer.numberOfTapsRequired = 1;
    [mtkView addGestureRecognizer:tapGestureRecognizer];
    
    doubleTapGestureRecognizer = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(onDoubleTap:)];
    doubleTapGestureRecognizer.numberOfTapsRequired = 2;
    [mtkView addGestureRecognizer:doubleTapGestureRecognizer];
    
    [tapGestureRecognizer requireGestureRecognizerToFail:doubleTapGestureRecognizer];
    
    auto scaleRecognizer = [[UIPinchGestureRecognizer alloc] initWithTarget:self action:@selector(onPinch:)];
    [mtkView addGestureRecognizer:scaleRecognizer];
    
    lightDirection = normalize(float2{1.f, -1.f});
    
    motionManager = [CMMotionManager new];
    motionManager.deviceMotionUpdateInterval = 1.0 / 60.0;
    
    if ([motionManager isDeviceMotionAvailable])
    {
        __weak Metal4Renderer* wSelf = self;
        
        // Start getting updates
        [motionManager startDeviceMotionUpdatesToQueue:[NSOperationQueue mainQueue]
                                          withHandler:^(CMDeviceMotion *motion, NSError *error) {
            if (error != nil)
            {
                return;
            }

            Metal4Renderer* self = wSelf;
            if (self == nil)
            {
                return;
            }
             
            float angle;
            
            const auto orientation = [UIDevice currentDevice].orientation;
            if (UIDeviceOrientationIsPortrait(orientation))
            {
                angle = motion.attitude.roll;
            }
            else
            {
                angle = motion.attitude.yaw;
            }
            
            self->lightDirection = { cosf(angle), sinf(angle) };
            
        }];
    }
    return self;
}

- (Uniforms*)uniforms
{
    return reinterpret_cast<Uniforms*>(uniformsBuffer.contents);
}

- (void)updateUniformsBuffer
{
    auto buf = [self uniforms];
    
    buf->viewportSize = viewportSize;
    
    constexpr float s = 3e1f;
    const float2 gradientScale { s / float(sdfGradientTexture.width), s / float(sdfGradientTexture.height) };
    
    buf->gradientScale = gradientScale;
    
    buf->lightDirection = lightDirection;
    
    _bubbleSet.update(*buf);
    
}

/// The system calls this method whenever the view changes orientation or size.
- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size
{
    // Update the viewport property to its new size,
    // which the renderer passes to the vertex shader.
    viewportSize.x = (simd_uint1)size.width;
    viewportSize.y = (simd_uint1)size.height;

    [self updateUniformsBuffer];
}

- (void)drawSDFs:(id<MTL4ComputeCommandEncoder>)computeEncoder
{
    [computeEncoder setComputePipelineState:drawSDFPipelineState];
    
    // Configure the encoder's argument table for the dispatch call.
    [computeEncoder setArgumentTable:argumentTable];

    // Bind the composite color (input) texture in the argument table.
    [argumentTable setTexture:sdfTexture.gpuResourceID
                      atIndex:ComputeTextureBindingIndexForSDF];

    [argumentTable setAddress:uniformsBuffer.gpuAddress
                      atIndex:BufferBindingIndexForUniforms];
    
    // Run the dispatch with the pipeline state and current state of the argument table.
    [computeEncoder dispatchThreadgroups:threadgroupCount
                   threadsPerThreadgroup:threadgroupSize];
}

- (void)drawSDFGradient:(id<MTL4ComputeCommandEncoder>)computeEncoder
{
    [computeEncoder setComputePipelineState:drawSDFGradientPipelineState];
    
    // Configure the encoder's argument table for the dispatch call.
    [computeEncoder setArgumentTable:argumentTable];

    // Bind the composite color (input) texture in the argument table.
    [argumentTable setTexture:sdfTexture.gpuResourceID
                      atIndex:ComputeTextureBindingIndexForSDF];
    
    [argumentTable setTexture:sdfGradientTexture.gpuResourceID
                      atIndex:ComputeTextureBindingIndexForGradientSDF];
    
    // Run the dispatch with the pipeline state and current state of the argument table.
    [computeEncoder dispatchThreadgroups:threadgroupCount
                   threadsPerThreadgroup:threadgroupSize];
}

- (void)encodeComputePassWithEncoder:(id<MTL4ComputeCommandEncoder>)computeEncoder
{
    // Add a barrier that pauses the dispatch stage of the compute pass
    // from starting until the copy commands finish during their blit stage.
    [computeEncoder barrierAfterEncoderStages:MTLStageBlit
                          beforeEncoderStages:MTLStageDispatch
                            visibilityOptions:MTL4VisibilityOptionDevice];

    [self drawSDFs:computeEncoder];
    [self drawSDFGradient:computeEncoder];
}

- (void)encodeRenderPassWithEncoder:(id<MTL4RenderCommandEncoder>)renderEncoder
{
    // Add a barrier that tells the GPU to wait for any previous dispatch kernels
    // to finish before running any subsequent vertex stages.
    [renderEncoder barrierAfterQueueStages:MTLStageDispatch
                              beforeStages:MTLStageVertex
                         visibilityOptions:MTL4VisibilityOptionDevice];

    // Configure the view-port with the size of the drawable region.
    MTLViewport viewPort;
    viewPort.originX = 0.0;
    viewPort.originY = 0.0;
    viewPort.width = (double)viewportSize.x;
    viewPort.height = (double)viewportSize.y;
    viewPort.znear = 0.0;
    viewPort.zfar = 1.0;

    [renderEncoder setViewport:viewPort];

    // Configure the encoder with the renderer's main pipeline state.
    [renderEncoder setRenderPipelineState:renderPipelineState];

    // Set the encoder's argument table.
    [renderEncoder setArgumentTable:argumentTable
                           atStages:MTLRenderStageVertex | MTLRenderStageFragment];

    // Bind the buffer with the triangle data to the argument table.
    [argumentTable setAddress:vertexDataBuffer.gpuAddress
                      atIndex:BufferBindingIndexForVertexData];

    // Bind the buffer with the viewport's size to the argument table.
    [argumentTable setAddress:uniformsBuffer.gpuAddress
                      atIndex:BufferBindingIndexForUniforms];

    // Bind the color composite texture.
    [argumentTable setTexture:backgroundImageTexture.gpuResourceID
                      atIndex:RenderTextureBindingIndex];
    
    [argumentTable setTexture:sdfGradientTexture.gpuResourceID
                      atIndex:SDFGradientTextureBindingIndex];
    
    // Draw the first rectangle with the color composite texture.
    const NSUInteger firstRectangleOffset = 0;
    const NSUInteger rectangleVertexCount = 6;
    [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle
                      vertexStart:firstRectangleOffset
                      vertexCount:rectangleVertexCount];
}

/// Draws a frame of content to a view's drawable.
/// - Parameter view: A view with a drawable that the renderer draws into.
- (void)drawInMTKView:(nonnull MTKView *)view
{
    // Retrieve the view's drawable.
    id<CAMetalDrawable> drawable = view.currentDrawable;

    if (nil == drawable)
    {
        NSLog(@"The view doesn't have an available drawable at this time.");
        return;
    }
    
    [self updateUniformsBuffer];

    // Get the render pass descriptor from the view's drawable instance.
    MTL4RenderPassDescriptor *renderPassDescriptor = view.currentMTL4RenderPassDescriptor;

    if (nil == renderPassDescriptor)
    {
        NSLog(@"The view doesn't have a render pass descriptor for Metal 4.");
        return;
    }

    // Increment the frame number for this frame.
    frameNumber += 1;

    // Make a string with the current frame number,
    NSString *forFrameString = [NSString stringWithFormat:@" for frame: %llu", frameNumber];

    if (frameNumber >= kMaxFramesInFlight) {
        // Wait for the GPU to finish rendering the frame that's
        // `kMaxFramesInFlight` before this one, and then proceed to the next step.
        uint64_t previousValueToWaitFor = frameNumber - kMaxFramesInFlight;
        [sharedEvent waitUntilSignaledValue:previousValueToWaitFor
                                  timeoutMS:10];
    }

    /// The array index for this frame's resources.
    uint32_t frameIndex = frameNumber % kMaxFramesInFlight;

    /// An allocator that's next in the rotation for this frame.
    id<MTL4CommandAllocator> frameAllocator = commandAllocators[frameIndex];

    // Prepare to use or reuse the allocator by resetting it.
    [frameAllocator reset];

    // Reset the command buffer for the new frame.
    [commandBuffer beginCommandBufferWithAllocator:frameAllocator];

    // Assign the command buffer a unique label for this frame.
    commandBuffer.label = [@"Command buffer" stringByAppendingString:forFrameString];

    // === Compute pass ===
    // Create a compute encoder from the command buffer.
    id<MTL4ComputeCommandEncoder> computeEncoder;
    computeEncoder = [commandBuffer computeCommandEncoder];

    // Assign the compute encoder a unique label for this frame.
    computeEncoder.label = [@"Compute encoder" stringByAppendingString:forFrameString];

    // Encode a compute pass that copies the color textures and creates the grayscale texture.
    [self encodeComputePassWithEncoder:computeEncoder];

    // Mark the end of the compute pass.
    [computeEncoder endEncoding];

    // === Render pass ===
    // Create a render encoder from the command buffer.
    id<MTL4RenderCommandEncoder> renderEncoder =
    [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];

    // Assign the render encoder a unique label for this frame.
    renderEncoder.label = [@"Render encoder" stringByAppendingString:forFrameString];

    // Encode a render pass that draws a rectangle each of the composite textures.
    [self encodeRenderPassWithEncoder:renderEncoder];

    // Mark the end of the render pass.
    [renderEncoder endEncoding];

    // Finalize the command buffer.
    [commandBuffer endCommandBuffer];

    // === Submit passes to the GPU ===
    // Wait until the drawable is ready for rendering.
    [commandQueue waitForDrawable:drawable];

    // Submit the command buffer to the GPU.
    [commandQueue commit:&commandBuffer count:1];

    // Notify the drawable when the GPU finishes running the passes in the command buffer.
    [commandQueue signalDrawable:drawable];

    // Show the final result of this frame on the display.
    [drawable present];

    // Signal when the GPU finishes rendering this frame with a shared event.
    [commandQueue signalEvent:sharedEvent value:frameNumber];
}


- (float2)pointInSDFSpace:(float2)pos
{
    const CGSize viewSiz = view.bounds.size;
    
    const float2 viewSize { float(viewSiz.width), float(viewSiz.height) };
    const float2 textureSize { float(backgroundImageTexture.width), float(backgroundImageTexture.height) };
    
    const float2 scale = viewSize / textureSize;
    const float s = std::min(scale.x, scale.y);
    
    const float displayedTextureWidth = textureSize.x * s;
    const float displayedTextureHeight = displayedTextureWidth * (textureSize.y / textureSize.x);
    const float2 displayedTextureSize { displayedTextureWidth, displayedTextureHeight };
    const float2 displayedTextureOrigin = (viewSize - displayedTextureSize) * 0.5f;
    
    float2 p = pos - displayedTextureOrigin;
    p /= s;
    
    return p;
}

- (Bubble*)pick:(float2)pos
{
    const auto p = [self pointInSDFSpace:pos];
    return _bubbleSet.pick(p);
}

- (void)onPan:(UIPanGestureRecognizer*)recognizer
{
    const auto p = [recognizer locationInView:view];
    const float2 pos { float(p.x), float(p.y) };
    const auto ptInSDFSpace = [self pointInSDFSpace:pos];
    
    switch(recognizer.state)
    {
        case UIGestureRecognizerStateBegan:
        {
            if (auto bubble = [self pick:pos])
            {
                _bubbleSet.setSelection(*bubble, ptInSDFSpace);
            }
            else
            {
                _bubbleSet.clearSelection();
            }
            
            break;
        }
            
        case UIGestureRecognizerStateChanged:
        {
            _bubbleSet.moveSelection(ptInSDFSpace);
            break;
        }
            
        case UIGestureRecognizerStateEnded:
        case UIGestureRecognizerStateCancelled:
        {
            _bubbleSet.clearSelection();
            break;
        }
            
        default: break;
    }
}

class CPUTextureAccessor final
{
public:
    CPUTextureAccessor(id<MTLTexture> texture, uint2 gridId)
    : _texture(texture), _gridId(gridId), _value(std::make_shared<float>(0.f))
    {}
    
    CPUTextureAccessor(const CPUTextureAccessor&) = default;
    
    void write(float v)
    {
        *_value = v;
    }
    
    bool isValid() const
    {
        return (_gridId.x < _texture.width) && (_gridId.y < _texture.height);
    }
    
    float2 position() const
    {
        return { float(_gridId.x), float(_gridId.y) };
    }
    
    float value() const
    {
        return *_value;
    }
    
private:
    id<MTLTexture> _texture;
    uint2 _gridId;
    
    std::shared_ptr<float> _value;
};

- (void)onTap:(UITapGestureRecognizer*)recognizer
{
    if (recognizer.state == UIGestureRecognizerStateRecognized)
    {
        auto uniforms = [self uniforms];
        
        const CGPoint ptView = [recognizer locationInView:view];
        const float2 ptSDF = [self pointInSDFSpace: float2{ float(ptView.x), float(ptView.y) }];
        
        const uint2 pos { uint32_t(ptSDF.x), uint32_t(ptSDF.y) };
        
        CPUTextureAccessor accessor { backgroundImageTexture, pos };
        computeAndDrawSDF(accessor, uniforms);
        
        const auto value = accessor.value();
        NSLog(@"value [%1.2f]", value);
    }
}

- (void)onDoubleTap:(UITapGestureRecognizer*)recognizer
{
    if (recognizer.state == UIGestureRecognizerStateRecognized)
    {
        const CGPoint ptView = [recognizer locationInView:view];
        const float2 ptSDF = [self pointInSDFSpace: float2{ float(ptView.x), float(ptView.y) }];
        
        if (auto bubble = _bubbleSet.pick(ptSDF))
        {
            _bubbleSet.remove(*bubble);
        }
        else
        {
            // add
            _bubbleSet.add(ptSDF, 100.f);
        }
    }
}

- (void)onPinch:(UIPinchGestureRecognizer*)recognizer
{
    const auto p = [recognizer locationInView:view];
    const float2 pos { float(p.x), float(p.y) };
    const auto posInSDFSpace = [self pointInSDFSpace:pos];
    
    switch(recognizer.state)
    {
        case UIGestureRecognizerStateBegan:
        {
            if (auto bubble = _bubbleSet.pick(posInSDFSpace))
            {
                _bubbleSet.setSelection(*bubble, posInSDFSpace);
            }
            else
            {
                _bubbleSet.clearSelection();
            }
            break;
        }
            
        case UIGestureRecognizerStateChanged:
        {
            _bubbleSet.rescaleSelection(recognizer.scale);
            break;
        }
            
        case UIGestureRecognizerStateEnded:
        case UIGestureRecognizerStateCancelled:
        {
            _bubbleSet.clearSelection();
            break;
        }
            
        default: break;
    }
}

@end
