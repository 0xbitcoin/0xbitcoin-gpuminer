#ifndef _CUDASOLVER_H_
#define _CUDASOLVER_H_

#include <atomic>
#include <mutex>
#include <string>
#include <vector>


class CUDASolver
{
public:
  typedef std::vector<uint8_t> bytes_t;

  static const unsigned short ADDRESS_LENGTH = 20u;
  static const unsigned short UINT256_LENGTH = 32u;

public:
  CUDASolver() noexcept;

public:
  void setAddress(std::string const& addr);

  void setChallenge(std::string const& chal);

  void setTarget(std::string const& target);

  void setBlocksize(int  size);

  void setThreadsize(int  size);

  void init();

public:
  bytes_t findSolution( );
  void stopFinding( );

public:
  static void hexToBytes(std::string const& hex, bytes_t& bytes);

  static std::string bytesToString(bytes_t const& buffer);

  static  std::string hexStr(  char* data, int len);

  static bool lte(bytes_t const& left, bytes_t const& right);

  bool requiresRestart();


private:
  //void updateBuffer();

  void updateGPULoop();


  void hash(bytes_t const& solution, bytes_t& digest);

private:
  std::string s_challenge;
  std::string s_target;
  bytes_t m_address;
  bytes_t m_challenge;
  bytes_t m_target, m_target_tmp;
  bytes_t m_buffer, m_buffer_tmp;
  std::mutex m_buffer_mutex;
  std::mutex m_target_mutex;
  std::atomic<bool> m_buffer_ready;
  std::atomic<bool> m_target_ready;

  std::atomic<bool> m_updated_gpu_inputs;
public:
  static std::atomic<uint32_t> hashes;
};

#endif // !_SOLVER_H_
