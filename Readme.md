# DEPIR based on new Algebraic Homomorphic Encryption

This is our implementation of DEPIR using an improved ASHE scheme, as presented in our paper "On Algebraic Homomorphic Encryption and its Applications to Doubly-Efficient PIR" ([https://ia.cr/2024/1307](https://ia.cr/2024/1307)).

Unfortunately, it currently uses the asynchronous read API of Windows, thus does not run on Linux as is.
To run the code, adjust the constants `m`, `PREPROCESSOR_PATH`, `DATASTRUCTURE_PATH` in `src/main.rs`, then use `cargo`.
You probably have to compile the GPU-accelerated preprocessor first (in `poly-batch-eval`), although the test will run without it.