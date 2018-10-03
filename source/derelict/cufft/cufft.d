module derelict.cufft.cufft;

import derelict.util.loader;

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
  CUFFT_INVALID_SIZE   = 0x8
};
    
// CUFFT defines and supports the following data types

// cufftHandle is a handle type used to store and access CUFFT plans.
alias cufftHandle = uint;

// cufftReal is a single-precision, floating-point real data type.
alias cufftReal = float;

// cufftComplex is a single-precision, floating-point complex data type that 
// consists of interleaved real and imaginary components.
// typedef float cufftComplex[2];
import std.complex;
alias cufftComplex = Complex!(float);

// CUFFT transform directions 
enum CUFFT_FORWARD = -1; // Forward FFT
enum CUFFT_INVERSE =  1; // Inverse FFT

// CUFFT supports the following transform types 
enum cufftType {
  CUFFT_R2C = 0x2a, // Real to Complex (interleaved)
  CUFFT_C2R = 0x2c, // Complex (interleaved) to Real
  CUFFT_C2C = 0x29  // Complex to Complex, interleaved
};

extern(System) @nogc nothrow
{
    alias da_cufftPlan1d = cufftResult function(cufftHandle*, int, cufftType, int);
    alias da_cufftPlan2d = cufftResult function(cufftHandle*, int, int, cufftType);
    alias da_cufftPlan3d = cufftResult function(cufftHandle*, int, int, int, cufftType);
    alias da_cufftDestroy = cufftResult function(cufftHandle);
    alias da_cufftExecC2C = cufftResult function(cufftHandle, cufftComplex*, cufftComplex*, int);
    alias da_cufftExecR2C = cufftResult function(cufftHandle, cufftReal*, cufftComplex*);
    alias da_cufftExecC2R = cufftResult function(cufftHandle, cufftComplex*, cufftReal*);
}

__gshared
{
    da_cufftPlan1d cufftPlan1d;
    da_cufftPlan2d cufftPlan2d;
    da_cufftPlan3d cufftPlan3d;
    da_cufftDestroy cufftDestroy;
    da_cufftExecC2C cufftExecC2C;
    da_cufftExecR2C cufftExecR2C;
    da_cufftExecC2R cufftExecC2R;
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
