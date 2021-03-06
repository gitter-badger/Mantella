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
    class GallaghersGaussian21hiPeaksFunction : public BlackBoxOptimisationBenchmark {
     public:
      explicit GallaghersGaussian21hiPeaksFunction(
          const arma::uword numberOfDimensions);

      virtual void setRotationQ(
          const arma::Mat<double>& rotationQ);

      virtual void setLocalParameterConditionings(
          const arma::Mat<double>& localParameterConditionings);

      virtual void setLocalParameterTranslations(
          const arma::Mat<double>& localParameterTranslations);

      std::string toString() const override;
#if defined(SUPPORT_MPI)
      std::vector<double> serialise() const;
      void deserialise(
          std::vector<double> serialisedOptimisationProblem);
#endif

     protected:
      const arma::Col<double>::fixed<21> weight_;

      arma::Mat<double> rotationQ_;
      arma::Mat<double> localParameterConditionings_;
      arma::Mat<double> localParameterTranslations_;

      arma::Mat<double> getRandomLocalParameterConditionings() const;

      arma::Mat<double> getRandomLocalParameterTranslations() const;

      double getObjectiveValueImplementation(
          const arma::Col<double>& parameter) const override;
    };
  }
}
