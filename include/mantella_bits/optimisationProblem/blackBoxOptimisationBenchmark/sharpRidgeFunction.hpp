#pragma once
#include "mantella_bits/config.hpp" // IWYU pragma: keep

// C++ standard library
#include <string>
#if defined(SUPPORT_MPI) // IWYU pragma: keep
#include <vector>
#endif

// Armadillo
#include <armadillo>

// Mantella
#include "mantella_bits/optimisationProblem/blackBoxOptimisationBenchmark.hpp"

namespace mant {
  namespace bbob {
    class SharpRidgeFunction : public BlackBoxOptimisationBenchmark {
     public:
      explicit SharpRidgeFunction(
          const arma::uword numberOfDimensions);

      void setRotationQ(
          const arma::Mat<double>& rotationQ);

      std::string toString() const override;
#if defined(SUPPORT_MPI)
      std::vector<double> serialise() const;
      void deserialise(
          std::vector<double> serialisedOptimisationProblem);
#endif

     protected:
      const arma::Col<double> parameterConditioning_;

      arma::Mat<double> rotationQ_;

      double getObjectiveValueImplementation(
          const arma::Col<double>& parameter) const override;
    };
  }
}
