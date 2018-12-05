module derelict.cufft.cufft;

import derelict.util.loader;

import std.complex;

private
{
    import derelict.util.system;

    static if(Derelict_OS_Windows)
        enum libNames = "libcufft.dll";
    else static if (Derelict_OS_Mac)
        enum libNames = "libcufft.dylib";
    else static if (Derelict_OS_Linux)
    {
        version(X86)
            enum libNames = "libcufft.so";
        else version(X86_64)
            enum libNames = "libcufft.so";
        else
            static assert(0, "Need to implement cuFFT libNames for this arch.");
    }
    else
        static assert(0, "Need to implement cuFFT libNames for this operating system.");
}

// CUFFT API function return values 
enum cufftResult {
  CUFFT_SUCCESS        = 0x0,
  CUFFT_INVALID_PLAN   = 0x1,
  CUFFT_ALLOC_FAILED   = 0x2,
  CUFFT_INVALID_TYPE   = 0x3,
  CUFFT_INVALID_VALUE  = 0x4,
  CUFFT_INTERNAL_ERROR = 0x5,
  CUFFT_EXEC_FAILED    = 0x6,
  CUFFT_SETUP_FAILED   = 0x7,
  CUFFT_INVALID_SIZE   = 0x8,
  CUFFT_UNALIGNED_DATA = 0x9,
  CUFFT_INCOMPLETE_PARAMETER_LIST = 0xA,
  CUFFT_INVALID_DEVICE = 0xB,
  CUFFT_PARSE_ERROR = 0xC,
  CUFFT_NO_WORKSPACE = 0xD,
  CUFFT_NOT_IMPLEMENTED = 0xE,
  CUFFT_LICENSE_ERROR = 0x0F,
  CUFFT_NOT_SUPPORTED = 0x10
}

enum MAX_CUFFT_ERROR = 0x11;
    
// CUFFT defines and supports the following data types

// cufftReal is a single-precision, floating-point real data type.
// cufftDoubleReal is a double-precision, real data type.
alias cufftReal = float;
alias cufftDoubleReal = double;

// cufftComplex is a single-precision, floating-point complex data type that 
// consists of interleaved real and imaginary components.
// cufftDoubleComplex is the double-precision equivalent.
alias cufftComplex = Complex!(cufftReal);
alias cufftDoubleComplex = Complex!(cufftDoubleReal);

// CUFFT transform directions 
enum CUFFT_FORWARD = -1; // Forward FFT
enum CUFFT_INVERSE =  1; // Inverse FFT

// CUFFT supports the following transform types 
enum cufftType {
  CUFFT_R2C = 0x2a,     // Real to Complex (interleaved)
  CUFFT_C2R = 0x2c,     // Complex (interleaved) to Real
  CUFFT_C2C = 0x29,     // Complex to Complex, interleaved
  CUFFT_D2Z = 0x6a,     // Double to Double-Complex
  CUFFT_Z2D = 0x6c,     // Double-Complex to Double
  CUFFT_Z2Z = 0x69      // Double-Complex to Double-Complex
};

// CUFFT supports the following data layouts
enum cufftCompatibility {
    CUFFT_COMPATIBILITY_FFTW_PADDING    = 0x01    // The default value
}

enum CUFFT_COMPATIBILITY_DEFAULT = cufftCompatibility.CUFFT_COMPATIBILITY_FFTW_PADDING;

//
// structure definition used by the shim between old and new APIs
//
enum MAX_SHIM_RANK = 3;

// cufftHandle is a handle type used to store and access CUFFT plans.
alias cufftHandle = int;

alias cudaStream_t = void*;

extern(System) @nogc nothrow
{
    alias da_cufftPlan1d =
        cufftResult function(cufftHandle* plan,
                             int nx,
                             cufftType type,
                             int batch);

    alias da_cufftPlan2d = 
        cufftResult function(cufftHandle* plan,
                             int nx, int ny,
                             cufftType type);

    alias da_cufftPlan3d =
        cufftResult function(cufftHandle* plan,
                             int nx, int ny, int nz,
                             cufftType type);

    alias da_cufftPlanMany =
        cufftResult function(cufftHandle* plan,
                             int rank,
                             int *n,
                             int *inembed, int istride, int idist,
                             int *onembed, int ostride, int odist,
                             cufftType type,
                             int batch);
 
    alias da_cufftMakePlan1d =
        cufftResult function(cufftHandle plan, 
                             int nx, 
                             cufftType type, 
                             int batch,
                             size_t *workSize);

    alias da_cufftMakePlan2d =
        cufftResult function(cufftHandle plan, 
                             int nx, int ny,
                             cufftType type,
                             size_t *workSize);

    alias da_cufftMakePlan3d =
        cufftResult function(cufftHandle plan, 
                             int nx, int ny, int nz, 
                             cufftType type,
                             size_t *workSize);

    alias da_cufftMakePlanMany =
        cufftResult function (cufftHandle plan,
                              int rank,
                              int *n,
                              int *inembed, int istride, int idist,
                              int *onembed, int ostride, int odist,
                              cufftType type,
                              int batch,
                              size_t *workSize);
                                      
    alias da_cufftMakePlanMany64 =
        cufftResult function (cufftHandle plan, 
                              int rank, 
                              long *n,
                              long *inembed, 
                              long istride, 
                              long idist,
                              long *onembed, 
                              long ostride, long odist,
                              cufftType type, 
                              long batch,
                              size_t * workSize);

    alias da_cufftGetSizeMany64 =
        cufftResult function (cufftHandle plan,
                              int rank,
                              long *n,
                              long *inembed, 
                              long istride, long idist,
                              long *onembed, 
                              long ostride, long odist,
                              cufftType type,
                              long batch,
                              size_t *workSize);

    alias da_cufftEstimate1d =
        cufftResult function (int nx, 
                              cufftType type, 
                              int batch,
                              size_t *workSize);

    alias da_cufftEstimate2d =
        cufftResult function (int nx, int ny,
                              cufftType type,
                              size_t *workSize);

    alias da_cufftEstimate3d =
        cufftResult function (int nx, int ny, int nz, 
                              cufftType type,
                              size_t *workSize);

    alias da_cufftEstimateMany =
        cufftResult function (int rank,
                              int *n,
                              int *inembed, int istride, int idist,
                              int *onembed, int ostride, int odist,
                              cufftType type,
                              int batch,
                              size_t *workSize);
                                     
    alias da_cufftCreate =
        cufftResult function (cufftHandle * handle);                                     

    alias da_cufftGetSize1d =
        cufftResult function (cufftHandle handle, 
                              int nx, 
                              cufftType type, 
                              int batch,
                              size_t *workSize );
                                                                         
    alias da_cufftGetSize2d =
        cufftResult function (cufftHandle handle, 
                              int nx, int ny,
                              cufftType type,
                              size_t *workSize);

    alias da_cufftGetSize3d =
        cufftResult function (cufftHandle handle,
                              int nx, int ny, int nz, 
                              cufftType type,
                              size_t *workSize);

    alias da_cufftGetSizeMany =
        cufftResult function (cufftHandle handle, 
                              int rank, int *n,
                              int *inembed, int istride, int idist,
                              int *onembed, int ostride, int odist,
                              cufftType type, int batch, size_t *workArea);

    alias da_cufftGetSize =
        cufftResult function (cufftHandle handle, size_t *workSize);
                                               
    alias da_cufftSetWorkArea =
        cufftResult function (cufftHandle plan, void *workArea);

    alias da_cufftSetAutoAllocation =
        cufftResult function (cufftHandle plan, int autoAllocate);

    alias da_cufftExecC2C = 
        cufftResult function (cufftHandle plan, 
                              cufftComplex *idata,
                              cufftComplex *odata,
                              int direction);

    alias da_cufftExecR2C =
        cufftResult function (cufftHandle plan, 
                              cufftReal *idata,
                              cufftComplex *odata);

    alias da_cufftExecC2R =
        cufftResult function (cufftHandle plan, 
                              cufftComplex *idata,
                              cufftReal *odata);

    alias da_cufftExecZ2Z =
        cufftResult function (cufftHandle plan, 
                              cufftDoubleComplex *idata,
                              cufftDoubleComplex *odata,
                              int direction);

    alias da_cufftExecD2Z =
        cufftResult function (cufftHandle plan, 
                              cufftDoubleReal *idata,
                              cufftDoubleComplex *odata);

    alias da_cufftExecZ2D =
        cufftResult function (cufftHandle plan, 
                              cufftDoubleComplex *idata,
                              cufftDoubleReal *odata);
                                  

    // utility functions
    alias da_cufftSetStream = 
        cufftResult function (cufftHandle plan,
                              cudaStream_t stream);

    // This function was removed on CUDA 9.2
    /*
    alias da_cufftSetCompatibilityMode =
        cufftResult function (cufftHandle plan,
                              cufftCompatibility mode);
    */

    alias da_cufftDestroy =
        cufftResult function (cufftHandle plan);

    alias da_cufftGetVersion = 
        cufftResult function (int *_version);

    //alias da_cufftGetProperty =
    //    cufftResult function (libraryPropertyType type,
    //                          int *value);
}

__gshared
{
    da_cufftPlan1d cufftPlan1d;
    da_cufftPlan2d cufftPlan2d; 
    da_cufftPlan3d cufftPlan3d;
    da_cufftPlanMany cufftPlanMany;
    da_cufftMakePlan1d cufftMakePlan1d;
    da_cufftMakePlan2d cufftMakePlan2d;
    da_cufftMakePlan3d cufftMakePlan3d;
    da_cufftMakePlanMany cufftMakePlanMany;
    da_cufftMakePlanMany64 cufftMakePlanMany64;
    da_cufftGetSizeMany64 cufftGetSizeMany64;
    da_cufftEstimate1d cufftEstimate1d;
    da_cufftEstimate2d cufftEstimate2d;
    da_cufftEstimate3d cufftEstimate3d;
    da_cufftEstimateMany cufftEstimateMany;
    da_cufftCreate cufftCreate;
    da_cufftGetSize1d cufftGetSize1d;
    da_cufftGetSize2d cufftGetSize2d;
    da_cufftGetSize3d cufftGetSize3d;
    da_cufftGetSizeMany cufftGetSizeMany;
    da_cufftGetSize cufftGetSize;
    da_cufftSetWorkArea cufftSetWorkArea;
    da_cufftSetAutoAllocation cufftSetAutoAllocation;
    da_cufftExecC2C cufftExecC2C;
    da_cufftExecR2C cufftExecR2C;
    da_cufftExecC2R cufftExecC2R;
    da_cufftExecZ2Z cufftExecZ2Z;
    da_cufftExecD2Z cufftExecD2Z;
    da_cufftExecZ2D cufftExecZ2D;
    da_cufftSetStream cufftSetStream;
    //da_cufftSetCompatibilityMode cufftSetCompatibilityMode;
    da_cufftDestroy cufftDestroy;
    da_cufftGetVersion cufftGetVersion;

}

class DerelictCUFFTLoader : SharedLibLoader
{
    protected
    {
        override void loadSymbols()
        {
            bindFunc(cast(void**)&cufftPlan1d, "cufftPlan1d");
            bindFunc(cast(void**)&cufftPlan2d, "cufftPlan2d");
            bindFunc(cast(void**)&cufftPlan3d, "cufftPlan3d");
            bindFunc(cast(void**)&cufftDestroy, "cufftDestroy");
            bindFunc(cast(void**)&cufftExecC2C, "cufftExecC2C");
            bindFunc(cast(void**)&cufftExecR2C, "cufftExecR2C");
            bindFunc(cast(void**)&cufftExecC2R, "cufftExecC2R");
            bindFunc(cast(void**)&cufftPlan1d, "cufftPlan1d");
            bindFunc(cast(void**)&cufftPlan2d, "cufftPlan2d"); 
            bindFunc(cast(void**)&cufftPlan3d, "cufftPlan3d");
            bindFunc(cast(void**)&cufftPlanMany, "cufftPlanMany");
            bindFunc(cast(void**)&cufftMakePlan1d, "cufftMakePlan1d");
            bindFunc(cast(void**)&cufftMakePlan2d, "cufftMakePlan2d");
            bindFunc(cast(void**)&cufftMakePlan3d, "cufftMakePlan3d");
            bindFunc(cast(void**)&cufftMakePlanMany, "cufftMakePlanMany");
            bindFunc(cast(void**)&cufftMakePlanMany64, "cufftMakePlanMany64");
            bindFunc(cast(void**)&cufftGetSizeMany64, "cufftGetSizeMany64");
            bindFunc(cast(void**)&cufftEstimate1d, "cufftEstimate1d");
            bindFunc(cast(void**)&cufftEstimate2d, "cufftEstimate2d");
            bindFunc(cast(void**)&cufftEstimate3d, "cufftEstimate3d");
            bindFunc(cast(void**)&cufftEstimateMany, "cufftEstimateMany");
            bindFunc(cast(void**)&cufftCreate, "cufftCreate");
            bindFunc(cast(void**)&cufftGetSize1d, "cufftGetSize1d");
            bindFunc(cast(void**)&cufftGetSize2d, "cufftGetSize2d");
            bindFunc(cast(void**)&cufftGetSize3d, "cufftGetSize3d");
            bindFunc(cast(void**)&cufftGetSizeMany, "cufftGetSizeMany");
            bindFunc(cast(void**)&cufftGetSize, "cufftGetSize");
            bindFunc(cast(void**)&cufftSetWorkArea, "cufftSetWorkArea");
            bindFunc(cast(void**)&cufftSetAutoAllocation, "cufftSetAutoAllocation");
            bindFunc(cast(void**)&cufftExecC2C, "cufftExecC2C");
            bindFunc(cast(void**)&cufftExecR2C, "cufftExecR2C");
            bindFunc(cast(void**)&cufftExecC2R, "cufftExecC2R");
            bindFunc(cast(void**)&cufftExecZ2Z, "cufftExecZ2Z");
            bindFunc(cast(void**)&cufftExecD2Z, "cufftExecD2Z");
            bindFunc(cast(void**)&cufftExecZ2D, "cufftExecZ2D");
            bindFunc(cast(void**)&cufftSetStream, "cufftSetStream");
            //bindFunc(cast(void**)&cufftSetCompatibilityMode, "cufftSetCompatibilityMode");
            bindFunc(cast(void**)&cufftDestroy, "cufftDestroy");
            bindFunc(cast(void**)&cufftGetVersion, "cufftGetVersion");
        }
    }

    public
    {
        this()
        {
            super(libNames);
        }
    }
}

__gshared DerelictCUFFTLoader DerelictCUFFT;

shared static this()
{
    DerelictCUFFT = new DerelictCUFFTLoader();
}
