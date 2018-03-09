#ifndef _CUDASHA3_H_
#define _CUDASHA3_H_

#include <atomic>
#include <mutex>
#include <string>
#include <vector>


class CUDASha3
{
public:

public:
  CUDASha3() noexcept;

public:

  void gpu_init();
  void setCudaBlocksize(int blocksize);
  void setCudaThreadsize(int threadsize);
  void runBenchmarks();
  char *read_in_messages();
  int gcd(int a, int b);

private:
  // updated message the gpu_init() function
  int clock_speed;
  int number_multi_processors;
  int number_blocks;
  int number_threads;
  int max_threads_per_mp;

  int num_messages;
  const int digest_size = 256;
  const int digest_size_bytes = digest_size / 8;
  const size_t str_length = 7;	//change for different sizes

  cudaEvent_t start, stop;

};

#endif // !_SOLVER_H_
