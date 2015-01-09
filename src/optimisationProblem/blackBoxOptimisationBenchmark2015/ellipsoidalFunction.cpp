#include <mantella_bits/optimisationProblem/blackBoxOptimisationBenchmark2015/ellipsoidalFunction.hpp>

// Cereal
#include <cereal/archives/json.hpp>
#include <cereal/types/polymorphic.hpp>

namespace mant {
  namespace bbob2015 {
    double EllipsoidalFunction::getObjectiveValueImplementation(
        const arma::Col<double>& parameter) const noexcept {
      return arma::dot(scaling_, arma::square(getOscillationTransformation(parameter - translation_)));
    }

    std::string EllipsoidalFunction::to_string() const noexcept {
      return "EllipsoidalFunction";
    }
  }
}

CEREAL_REGISTER_TYPE(mant::bbob2015::EllipsoidalFunction)
