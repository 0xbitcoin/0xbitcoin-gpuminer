

#include <chrono>
#include <random>
#include <thread>

//allow print out
#include <iostream>
#include <string.h>
using namespace std;

#include "hybridminer.h"

// What is this code doing ? defining a new instance of a class ?
HybridMiner::HybridMiner() noexcept :
  m_solvers(std::thread::hardware_concurrency()),
  m_threads(std::thread::hardware_concurrency()),
  m_solution(CPUSolver::UINT256_LENGTH),
  m_bSolutionFound(false),
  m_bExit(false)

{
}


HybridMiner::~HybridMiner()
{
  stop();

  // Wait for run() to terminate
  //  This is not very clean but it's the easiest portable way to
  //  exit gracefully if stop() has not been called before the destructor.
  std::this_thread::yield();
  for (auto&& thr : m_threads)
  {
    if (!thr.joinable())
      std::this_thread::sleep_for(std::chrono::milliseconds(50u));
  }
}

//set the hardware type to 'cpu' or 'gpu'
void HybridMiner::setHardwareType(std::string const& hardwareType)
{
  cout << "Setting hardware type: ";
  cout << (m_hardwareType = hardwareType);
}


void HybridMiner::setChallengeNumber(std::string const& challengeNumber)
{
  if(strcmp(m_hardwareType.c_str(), "cuda") == 0)
  {
     gpuSolver.setChallenge(challengeNumber);
  }else{
    set(&CPUSolver::setChallenge, challengeNumber);
  }
}

void HybridMiner::setDifficultyTarget(std::string const& difficultyTarget)
{
  if(strcmp(m_hardwareType.c_str(), "cuda") == 0)
  {
     gpuSolver.setTarget(difficultyTarget);
  }else{
    set(&CPUSolver::setTarget, difficultyTarget);
  }
}

void HybridMiner::setMinerAddress(std::string const& minerAddress)
{
  if(strcmp(m_hardwareType.c_str(), "cuda") == 0)
  {
    gpuSolver.setAddress(minerAddress);
  }else{
    set(&CPUSolver::setAddress, minerAddress);
  }
}

// This is a the "main" thread of execution
void HybridMiner::run()
{

  cout << "--Starting mining loop hardware type is: " << m_hardwareType.c_str();

  if(  strcmp(m_hardwareType.c_str(), "cuda") == 0   )
  {
    cout << "--Starting mining loop using CUDA-- \n";


    //fill me in ! run  gpu solver

     GPUSolver::bytes_t solution(CPUSolver::UINT256_LENGTH);


     //this is core dumping - maybe when you dont have a GPU?
     gpuSolver.init();


  //  solution = gpuSolver.findSolution(solution);
  //  cout << "--GPU returned a soln ! -- \n";
  //  solutionFound(solution);

  }else{
    cout << "--Starting mining loop using CPU-- \n";

    m_bExit = m_bSolutionFound = false;

    // These are the Solver threads
    for (size_t x = 0; x < m_threads.size(); ++x)
      m_threads[x] = std::thread([&, x] { this->thr_func(this->m_solvers[x]); });

    for (auto&& thr: m_threads)
      thr.join();

  }


}

void HybridMiner::stop()
{
  m_bExit = true;
}

void HybridMiner::thr_func(CPUSolver& solver)
{
  std::random_device r;
  std::mt19937_64 gen(r());
  std::uniform_int_distribution<> dist(0, 0xffffffff);

  CPUSolver::bytes_t solution(CPUSolver::UINT256_LENGTH);

  while (!m_bExit)
  {
    for (size_t i = 0; i < solution.size(); i += 4)
    {
      uint32_t const tmp = dist(gen);
      solution[i]     = static_cast<uint8_t> (tmp & 0x000000ff);
      solution[i + 1] = static_cast<uint8_t>((tmp & 0x0000ff00) >> 8);
      solution[i + 2] = static_cast<uint8_t>((tmp & 0x00ff0000) >> 16);
      solution[i + 3] = static_cast<uint8_t>((tmp & 0xff000000) >> 24);
    }

    if (solver.trySolution(solution))
    {
      solutionFound(solution);
      break;
    }
  }
}

// When this function terminates, the "main" thread run() should end
//  and the caller can check the solution()
void HybridMiner::solutionFound(CPUSolver::bytes_t const& solution)
{
  {
    std::lock_guard<std::mutex> g(m_solution_mutex);
    m_solution = solution;
    m_bSolutionFound = true;
  }

  stop();
}

//edit a variable within each of the solvers
void HybridMiner::set(void (CPUSolver::*fn)(std::string const&), std::string const& p)
{
  for (auto&& i : m_solvers)
    (i.*fn)(p);
}

std::string HybridMiner::solution() const
{
  return m_bSolutionFound ? ("0x" + CPUSolver::bytesToString(m_solution)) : std::string();
}