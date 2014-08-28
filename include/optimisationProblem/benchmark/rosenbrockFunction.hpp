#pragma once

#include <optimisationProblem/benchmark/benchmarkProblem.hpp>

namespace hop {
  class RosenbrockFunction : public BenchmarkProblem {
    public:
      RosenbrockFunction(const unsigned int& numberOfDimensions);

      void setTranslation(const arma::Col<double>& translation) override;

    protected:
      const double _max;

      double getObjectiveValueImplementation(const arma::Col<double>& parameter) const override;

      friend class cereal::access;
      RosenbrockFunction() = default;

      template<class T>
      void serialize(T& archive) {
        archive(cereal::make_nvp("benchmarkProblem", cereal::base_class<BenchmarkProblem>(this)));
        archive(CEREAL_NVP(_translation));
      }
  };
}
