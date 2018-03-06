/*
  Header file to declare prototypes

*/



#ifndef  _HYBRIDMINER_H_
#define  _HYBRIDMINER_H_

#include "cpusolver.h"

#include "cudasolver.h"

#include <thread>
#include <string.h>
using namespace std;

class HybridMiner
{
public:
  HybridMiner() noexcept;
  ~HybridMiner();

public:
  void setChallengeNumber(std::string const& challengeNumber);
  void setDifficultyTarget(std::string const& difficultyTarget);
  void setMinerAddress(std::string const& minerAddress);

  void setHardwareType(std::string const& hardwareType);
  const std::string getHardwareType();

public:
  void run();
  void stop();

  std::string solution() const;

private:
  void thr_func(CPUSolver& solver);

  void solutionFound(CPUSolver::bytes_t const& solution);

  //set a var in the solver !!
private:
  void set(void (CPUSolver::*fn)(std::string const&), std::string const& p);

private:
  std::vector<CPUSolver> m_solvers;
  std::vector<std::thread> m_threads;

  CUDASolver cudaSolver;


  std::mutex m_solution_mutex;

  CPUSolver::bytes_t m_solution; //make one for GPU ?

//  GPUSolver gpuSolver;

  bool m_bSolutionFound;

  std::string m_hardwareType;

  volatile bool m_bExit;
};

#endif // ! _CPUMINER_H_
