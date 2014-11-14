#pragma once

// C++ Standard Library
#include <array>
#include <vector>

// HOP
#include <hop_bits/optimisationAlgorithm/populationBasedAlgorithm/parallelAlgorithm.hpp>

namespace hop {
  class ParallelStandardParticleSwarmOptimisation2011 : public ParallelAlgorithm {
    public:
      explicit ParallelStandardParticleSwarmOptimisation2011(
          const std::shared_ptr<OptimisationProblem> optimisationProblem,
          const unsigned int& populationSize);

      ParallelStandardParticleSwarmOptimisation2011(const ParallelStandardParticleSwarmOptimisation2011&) = delete;
      ParallelStandardParticleSwarmOptimisation2011& operator=(const ParallelStandardParticleSwarmOptimisation2011&) = delete;

      void setNeighbourhoodProbability(
          const double& neighbourhoodProbability);
      void setAcceleration(
          const double& acceleration);
      void setLocalAttraction(
          const double& localAttraction);
      void setGlobalAttraction(
          const double& globalAttraction);
      void setCommunicationSteps(
          const unsigned int& communicationSteps);

      std::string to_string() const override;

    protected:
      double neighbourhoodProbability_;
      double acceleration_;
      double localAttraction_;
      double globalAttraction_;

      unsigned int communicationSteps_;

      void parallelOptimiseImplementation() override;
  };
}
