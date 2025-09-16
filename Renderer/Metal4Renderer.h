
#import <MetalKit/MetalKit.h>

/// A renderer for systems that support Metal 4 GPUs.
@interface Metal4Renderer : NSObject<MTKViewDelegate>

- (nonnull instancetype)initWithView:(nonnull MTKView *)mtkView;

@end
