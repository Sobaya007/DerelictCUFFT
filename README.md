DerelictCUFFT
============

A dynamic binding to [cuFFT][1] for the D Programming Language.

Please see the pages [Building and Linking Derelict][2] and [Using Derelict][3], in the Derelict documentation, for information on how to build DerelictCuFFT and load the cuFFT library at run time. In the meantime, here's some sample code.

```D
import derelict.cufft;

void main() {

    DerelictCUFFT.load();
    // Now cuFFT API functions can be called.

    ...
}
```

[1] https://developer.nvidia.com/cufft
[2]: http://derelictorg.github.io/compiling.html
[3]: http://derelictorg.github.io/using.html
