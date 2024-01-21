// GPU stage of approach to find 'Siege on Castle Steve Video'
// I used some code from Andrew.
// Started modding it on 2023-12-11
// Finished on: 2024-01-04
// by pablion

#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <sstream>
#include <iomanip>
#include <immintrin.h>
#include <cstdint>
#include <cuda_runtime.h>

#define ll long long int

using namespace std;

#define LENGTH 32768

__device__ __managed__ ll ret[1000000];
__device__ __managed__ int lX[1000000];
__device__ __managed__ int lY[1000000];
__device__ __managed__ int lZ[1000000];
__device__ __managed__  int numWritten;

__device__ __managed__ int useGPUs = 0;

std::string useOutput, useInput;

__device__ __managed__ int useHeight = 0;

typedef struct
{
	ll seed;
} javarand;

const ll multiplier = 0x5DEECE66Dll;

const ll addend = 0xBll;

const ll mask = (1ll << 48) - 1;

const float DOUBLE_UNIT = 1.0 / (1ll << 53);

__device__ int next(int bits, ll &seed) 
{
    seed = (seed * multiplier + addend) & mask;

	return (int)(seed >> (48 - bits));
}

__device__ ll nextSeed(ll &seed, ll multiplier, ll addend)
{
    seed = (seed * multiplier + addend) & mask;

	return seed;
}

__device__ int nextInt(int bound, ll &seed)
{
    int r = next(31, seed);

    int m = bound - 1;

    if ((bound & m) == 0)  // i.e., bound is a power of 2
    {
        r = (int)((bound * (long)r) >> 31);
    }
    else
    {
        for (int u = r; u - (r = u % bound) + m < 0; u = next(31, seed));
    }

    return r;
}

__device__ float nextDouble(ll &seed)
{
    return ((((ll)(next(26, seed)) << 27) + next(27, seed)) * DOUBLE_UNIT);
}

__device__ void fillNode(int nodes[])
{
    // Got it from in-game recreation of stone placements.

    int store_82 = 2 + 8 * (0 + 16 * 10); nodes[0] = store_82;
    int store_194 = 2 + 8 * (1 + 16 * 8); nodes[1] = store_194;
    int store_218 = 2 + 8 * (1 + 16 * 11); nodes[2] = store_218;
    int store_314 = 2 + 8 * (2 + 16 * 7); nodes[3] = store_314;
    int store_346 = 2 + 8 * (2 + 16 * 11); nodes[4] = store_346;
    int store_657 = 1 + 8 * (5 + 16 * 2); nodes[5] = store_657;
    int store_722 = 2 + 8 * (5 + 16 * 10); nodes[6] = store_722;
    int store_730 = 2 + 8 * (5 + 16 * 11); nodes[7] = store_730;
    int store_777 = 1 + 8 * (6 + 16 * 1); nodes[8] = store_777;
    int store_802 = 2 + 8 * (6 + 16 * 4); nodes[9] = store_802;
    int store_826 = 2 + 8 * (6 + 16 * 7); nodes[10] = store_826;
    int store_1001 = 1 + 8 * (7 + 16 * 13); nodes[11] = store_1001;
    int store_1025 = 1 + 8 * (8 + 16 * 0); nodes[12] = store_1025;
    int store_1129 = 1 + 8 * (8 + 16 * 13); nodes[13] = store_1129;
    int store_1257 = 1 + 8 * (9 + 16 * 13); nodes[14] = store_1257;
    int store_1386 = 2 + 8 * (10 + 16 * 13); nodes[15] = store_1386;
    int store_1425 = 1 + 8 * (11 + 16 * 2); nodes[16] = store_1425;
    int store_1513 = 1 + 8 * (11 + 16 * 13); nodes[17] = store_1513;
    int store_1641 = 1 + 8 * (12 + 16 * 13); nodes[18] = store_1641;
    int store_1642 = 2 + 8 * (12 + 16 * 13); nodes[19] = store_1642;
    int store_1761 = 1 + 8 * (13 + 16 * 12); nodes[20] = store_1761;
    int store_1873 = 1 + 8 * (14 + 16 * 10); nodes[21] = store_1873;
    int store_1874 = 2 + 8 * (14 + 16 * 10); nodes[22] = store_1874;
}

__device__ void lavaGen3(ll seed, ll currSeed, int x, int y, int z)
{
    ll nextseed = seed;

    int l = nextInt(4, nextseed) + 4;

	const int AMOUNT = 23;

	const int BUFFER_SIZE = 2048;

    int lastScoreX = 0, lastScoreZ = 0;

    bool aflag[BUFFER_SIZE] = {0};

    int nodes[AMOUNT];

    fillNode(nodes);

	ll n_seed = nextseed;

	bool init = false;

    for(int i1 = 0; i1 < l; ++i1)
    {
        float d = nextDouble(n_seed) * 6 + 3;
        float d1 = nextDouble(n_seed) * 4 + 2;
        float d2 = nextDouble(n_seed) * 6 + 3;
        float d3 = nextDouble(n_seed) * (16 - d - 2) + 1.0 + d / 2;
        float d4 = nextDouble(n_seed) * (8 - d1 - 4) + 2 + d1 / 2;
        float d5 = nextDouble(n_seed) * (16 - d2 - 2) + 1.0 + d2 / 2;
		
        for(int j4 = 1; j4 <= 15; ++j4)
        {
            for(int k4 = 1; k4 <= 15; ++k4)
            {
                for(int l4 = 1; l4 <= 7; ++l4)
                {
                    if(l4 != 4 && l4 != 5) // No need to check whole 16x16x8 box
                    {
                    	continue;
                    }

                    float d6 = ((float)j4 - d3) / (d / 2);
                    float d7 = ((float)l4 - d4) / (d1 / 2);
                    float d8 = ((float)k4 - d5) / (d2 / 2);

                    float d9 = d6 * d6 + d7 * d7 + d8 * d8;

                    if(d9 < 1)
                    {
                        aflag[(j4 * 16 + k4) * 8 + l4] = true;

                        if((j4 * 16 + k4) * 8 + (l4 + 1) < 2048) aflag[(j4 * 16 + k4) * 8 + (l4 + 1)] = true;
                        if((j4 * 16 + k4) * 8 + (l4 - 1) < 2048) aflag[(j4 * 16 + k4) * 8 + (l4 - 1)] = true;
                        
                        if(((j4 + 1) * 16 + k4) * 8 + l4 < 2048) aflag[((j4 + 1) * 16 + k4) * 8 + l4] = true;
                        if(((j4 - 1) * 16 + k4) * 8 + l4 < 2048) aflag[((j4 - 1) * 16 + k4) * 8 + l4] = true;
                        
                        if((j4 * 16 + (k4 + 1)) * 8 + l4 < 2048) aflag[(j4 * 16 + (k4 + 1)) * 8 + l4] = true;
                        if((j4 * 16 + (k4 - 1)) * 8 + l4 < 2048) aflag[(j4 * 16 + (k4 - 1)) * 8 + l4] = true;
                        
                        if(lastScoreX < j4)
                        {
                            if(aflag[((j4 + 1) * 16 + k4) * 8 + l4] && aflag[(j4 * 16 + k4) * 8 + l4])
                            {
                                lastScoreX = j4 + 1;
                            }
                        }
                    }
                }
            }
	    }
    }

	if(lastScoreX < 14) // Always something, right?
	{
		return;
	}

    for(int translateX = 0; translateX <= 2; translateX++)
    {
        for(int translateZ = 0; translateZ <= 2; translateZ++)
        {
            int goods = 0;
            
            for(int i = 0; i < AMOUNT; i++)
            {
                int Z = nodes[i] / (16 * 8);
                int X = (nodes[i] / 8) % 16;
                int Y = nodes[i] % 8;

                X += translateX;
                
                Y += 3;
                
                Z += translateZ;

                if((((X) * 16 + (Z)) * 8 + (Y)) < 2048 && (aflag[((X) * 16 + (Z)) * 8 + (Y)]))
                {
                    goods++;
                }
                
                if((((X) * 16 + (Z)) * 8 + (Y)) < 2048 && (!aflag[((X) * 16 + (Z)) * 8 + (Y)])) // Abort checking as soon as the pattern is invalid.
                {
                    break;
                }

                if(goods >= AMOUNT - 1)
                {
                    ret[numWritten] = currSeed;

                    lX[numWritten] = x;
                    
                    lY[numWritten] = y;

                    lZ[numWritten] = z;

                    atomicAdd(&numWritten, 1);

                    return;
                }
            }
        }
    }
}

__device__ void check3(ll seed, ll id)
{
    javarand random3;

    random3.seed = seed;

    ll cpy = random3.seed;

    if(nextInt(8, cpy) == 0)
    {
        ll saved_seed = cpy;
        
        int j1 = nextInt(16, saved_seed);

        int i5 = nextInt(nextInt(120, saved_seed) + 8, saved_seed);

        int j8 = nextInt(16, saved_seed);

        if(i5 < 64 || nextInt(10, saved_seed) == 0)
        {
            if(i5 >= 70 && i5 <= 120)
            {
                lavaGen3(saved_seed, cpy, 0, 0, 0);
            }
        }
    }
}

__global__ void m_kernel(ll gid)
{
    ll id = threadIdx.x + blockIdx.x * blockDim.x + gid;

    javarand random3;

    random3.seed = id;

    check3(random3.seed, id);
}

#define CHECK_GPU_ERR(code) gpuAssert((code), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s (code %d) %s %d\n", cudaGetErrorString(code), code, file, line);
        exit(code);
    }
}

cudaError_t do_work() {

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return cudaStatus;
	}

	ofstream logs(useInput);

	int threads_per_block = 512;
	int num_blocks = LENGTH; 

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
	printf("begin xyz\n");
	auto start = chrono::steady_clock::now();

    ll offset = 0; // stages, remember?

    ll NUM_ITERS = (1ll << 48) - offset;
    
	ofstream fout(useOutput);

    int GPU_COUNT = useGPUs;

	for(ll total = 0; total <= NUM_ITERS;)
    {
        for(int gpu_index = 0; gpu_index < GPU_COUNT; gpu_index++)
        {
            CHECK_GPU_ERR(cudaSetDevice(gpu_index));

            m_kernel <<< num_blocks, threads_per_block >>> (total + offset);

            total += num_blocks * threads_per_block;
        }

		if (total % (1L << 30) == 0)
        {	
            cudaDeviceSynchronize();

			for(int i = 0; i < numWritten; i++)
            {
				fout << ret[i] << endl;
			}

			numWritten = 0;

			auto end = chrono::steady_clock::now();
			ll time = (chrono::duration_cast<chrono::microseconds>(end - start).count());
			
			float eta = ((NUM_ITERS - total) / ((float)(total))) * ((float)time) / 3600.0 / 1000000.0; 

			logs << "doing " << (total + offset) << ", time taken us = " << time << ", ETA = " << eta << endl;
            double progress = ((double)total / ((double)NUM_ITERS)) * 100.0;
		}
	}
					
	fout.flush();
	
	fout.close();
	
	logs.close();
	
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda not sync: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
	auto end = chrono::steady_clock::now();
	cout << "time taken us =" << chrono::duration_cast<chrono::microseconds>(end - start).count() << endl;

	

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching big tree kernel!\n", cudaStatus);
	}

	return cudaStatus;
}

int main(int argc, char* argv[])
{
    useGPUs = atoi(argv[1]);

    useHeight = atoi(argv[2]); // Never used in practice.

    stringstream ss;

    ss << argv[3];

    stringstream ss2;

    ss2 << argv[4];

    useInput = ss.str();

    useOutput = ss2.str();

	cudaError_t cudaStatus = do_work();
	
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda failed!");
		return 1;
	}
	
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	
	return 0;
}
