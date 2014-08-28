#pragma once

#include <optimisationProblem/benchmark/benchmarkProblem.hpp>

namespace hop {
  class EllipsoidalFunctionRotated : public BenchmarkProblem {
    public:
      EllipsoidalFunctionRotated(const unsigned int& numberOfDimensions);

    protected:
      const arma::Col<double> _scaling;

      double getObjectiveValueImplementation(const arma::Col<double>& parameter) const override;

      friend class cereal::access;
      EllipsoidalFunctionRotated() = default;

      template<class T>
      void serialize(T& archive) {
        archive(cereal::make_nvp("benchmarkProblem", cereal::base_class<BenchmarkProblem>(this)));
        archive(CEREAL_NVP(_translation));
        archive(CEREAL_NVP(_rotationR));
      }
  };
}
