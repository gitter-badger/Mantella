// Catch
#include <catch.hpp>
#include <catchExtension.hpp>

// Mantella
#include <mantella>

class TestOptimisationAlgorithm : public mant::OptimisationAlgorithm {
  public:
    TestOptimisationAlgorithm(
        const std::shared_ptr<mant::OptimisationProblem> optimisationProblem)
        : mant::OptimisationAlgorithm(optimisationProblem) {
      }

    void optimiseImplementation() {

    }

    arma::Col<double> getRandomNeighbour(
        const arma::Col<double>& parameter,
        const arma::Col<double>& minimalDistance,
        const arma::Col<double>& maximalDistance){
      return mant::OptimisationAlgorithm::getRandomNeighbour(parameter,minimalDistance,maximalDistance);
    }

    std::string toString() const {
      return "TestOptimisationAlgorithm";
    }
};

TEST_CASE(
    "OptimisationAlgorithm") {
  SECTION("getRandomNeighbour") {
    arma::uword numberOfIterations = 10000;
    std::shared_ptr<mant::OptimisationProblem> optimisationProblem(new mant::bbob::SphereFunction(2));
    TestOptimisationAlgorithm testOptimisationAlgorithm(optimisationProblem);
    arma::Col<double> minimalDistance({1.0, 1.0});
    arma::Col<double> maximalDistance({5.0, 5.0});
    arma::Col<double> lastParameter({0.0, 0.0});

    arma::Col<double> values(numberOfIterations);

    for(arma::uword i = 0; i < numberOfIterations; i++) {
      arma::Col<double> newParameter(testOptimisationAlgorithm.getRandomNeighbour(lastParameter,minimalDistance,maximalDistance));
      values(i) = arma::norm(newParameter);
    }

    IS_UNIFORM(values, 1.0, 5.0);

  }
}
