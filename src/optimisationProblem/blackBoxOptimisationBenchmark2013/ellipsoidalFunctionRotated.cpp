#include <mantella_bits/optimisationProblem/blackBoxOptimisationBenchmark2013/ellipsoidalFunctionRotated.hpp>

// Cereal
#include <cereal/archives/json.hpp>
#include <cereal/types/polymorphic.hpp>

namespace mant {
  namespace bbob2013 {
    double EllipsoidalFunctionRotated::getObjectiveValueImplementation(
        const arma::Col<double>& parameter) const noexcept {
      return arma::dot(scaling_, arma::square(getOscillationTransformation(rotationR_ * (parameter - translation_))));
    }

    std::string EllipsoidalFunctionRotated::to_string() const noexcept {
      return "EllipsoidalFunctionRotated";
    }
  }
}

CEREAL_REGISTER_TYPE(mant::bbob2013::EllipsoidalFunctionRotated)
