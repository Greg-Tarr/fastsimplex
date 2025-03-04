#define WITH_METAL

#include <torch/extension.h>
#include <stdexcept>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

struct NoiseParams {
    int octaves;
    float persistence;
    float lacunarity;
    int seed;
};

static id<MTLComputePipelineState> getPipelineState(const char* kernelName) {
    static std::mutex pipelineMutex;
    static std::unordered_map<std::string, id<MTLComputePipelineState>> pipelineCache;

    std::lock_guard<std::mutex> lock(pipelineMutex);

    auto it = pipelineCache.find(kernelName);
    if (it != pipelineCache.end()) {
        return it->second;
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        throw std::runtime_error("Metal device not available");
    }

    NSString *currentFile = [NSString stringWithUTF8String:__FILE__];
    NSString *shaderPath = [[currentFile stringByDeletingLastPathComponent]
                            stringByAppendingPathComponent:@"_simplex.metal"];

    if (![[NSFileManager defaultManager] fileExistsAtPath:shaderPath]) {
        throw std::runtime_error(std::string("Metal shader file not found at: ") +
                               [shaderPath UTF8String]);
    }

    NSError *error = nil;
    NSString *shaderSource = [NSString stringWithContentsOfFile:shaderPath
                                                      encoding:NSUTF8StringEncoding
                                                         error:&error];
    if (!shaderSource) {
        throw std::runtime_error(std::string("Failed to load Metal shader source: ") +
                               [[error localizedDescription] UTF8String]);
    }

    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource
                                                 options:nil
                                                   error:&error];
    if (!library) {
        throw std::runtime_error("Failed to create Metal library");
    }

    id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:kernelName]];
    if (!function) {
        throw std::runtime_error("Failed to create Metal function");
    }

    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        throw std::runtime_error("Failed to create compute pipeline");
    }

    pipelineCache[kernelName] = pipeline;
    return pipeline;
}

// The Metal-based simplex noise function.
// (If the third tensor is empty, we run 2D; otherwise 3D.)
torch::Tensor simplex_metal(torch::Tensor x, torch::Tensor y, torch::Tensor z,
                        int octaves, float persistence, float lacunarity, int seed) {
  // Ensure tensors are contiguous and on CPU.
  x = x.contiguous();
  y = y.contiguous();
  if (z.numel() > 0) {
    z = z.contiguous();
  }
  auto count = x.numel();
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  // We allocate a 1D output tensor (we'll later reshape it to x's shape)
  torch::Tensor out = torch::empty({count}, options);

  @autoreleasepool {
    // Create the Metal device.
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      throw std::runtime_error("Metal device not available");
    }
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];

    // Pick the kernel function based on whether we have a z-coordinate.
    bool is3D = (z.numel() > 0);
    const char* kernelName = is3D ? "kernel_noise3d" : "kernel_noise2d";
    id<MTLComputePipelineState> pipelineState = getPipelineState(kernelName);


    // Create Metal buffers for the inputs and output.
    NSUInteger bufferLength = count * sizeof(float);
    id<MTLBuffer> xBuffer = [device newBufferWithBytes:x.data_ptr<float>()
                                                 length:bufferLength
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> yBuffer = [device newBufferWithBytes:y.data_ptr<float>()
                                                 length:bufferLength
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> outBuffer = [device newBufferWithBytesNoCopy:out.data_ptr<float>()
                                                        length:bufferLength
                                                       options:MTLResourceStorageModeShared
                                                   deallocator:nil];

    id<MTLBuffer> zBuffer = nil;
    if (is3D) {
      zBuffer = [device newBufferWithBytes:z.data_ptr<float>()
                                    length:bufferLength
                                   options:MTLResourceStorageModeShared];
    }

    // Pack the noise parameters.
    NoiseParams params;
    params.octaves     = octaves;
    params.persistence = persistence;
    params.lacunarity  = lacunarity;
    params.seed        = seed;
    id<MTLBuffer> paramsBuffer = [device newBufferWithBytes:&params
                                                     length:sizeof(NoiseParams)
                                                    options:MTLResourceStorageModeShared];

    // Create a command buffer and compute encoder.
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipelineState];

    if (is3D) {
      [encoder setBuffer:xBuffer offset:0 atIndex:0];
      [encoder setBuffer:yBuffer offset:0 atIndex:1];
      [encoder setBuffer:zBuffer offset:0 atIndex:2];
      [encoder setBuffer:outBuffer offset:0 atIndex:3];
      [encoder setBuffer:paramsBuffer offset:0 atIndex:4];
    } else {
      [encoder setBuffer:xBuffer offset:0 atIndex:0];
      [encoder setBuffer:yBuffer offset:0 atIndex:1];
      [encoder setBuffer:outBuffer offset:0 atIndex:2];
      [encoder setBuffer:paramsBuffer offset:0 atIndex:3];
    }

    // Dispatch one thread per noise sample.
    MTLSize gridSize = MTLSizeMake(count, 1, 1);
    NSUInteger threadGroupSize = pipelineState.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > count) {
      threadGroupSize = count;
    }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
  }

  return out.view(x.sizes());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("simplex", &simplex_metal, "Simplex noise (MPS)");
}
