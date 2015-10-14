// Catch
#include <catch.hpp>

// C++ Standard Library
#include <memory>

// Mantella
#include <mantella>

TEST_CASE(
    "HookeJeevesAlgorithm") {
  std::shared_ptr<mant::OptimisationProblem> optimisationProblem(new mant::bbob::SphereFunction(2));
  mant::HookeJeevesAlgorithm hookeJeevesAlgorithm(optimisationProblem);
  hookeJeevesAlgorithm.setMaximalNumberOfIterations(10000);

  SECTION(
      ".optimise") {
    // TODO
  }

  SECTION("Exception tests") {

    SECTION(
          "Throws an exception, if the InitialStepSize zero" ) {
      CHECK_THROWS_AS(hookeJeevesAlgorithm.setInitialStepSize({0, 0}), std::logic_error);
    }

    SECTION(
          "Throws an exception, if the InitialStepSize lower than zero" ) {
      CHECK_THROWS_AS(hookeJeevesAlgorithm.setInitialStepSize({-0.001, -8.5}), std::logic_error);
    }

    SECTION(
          "Throws an exception, if the size of InitialStepSize is not equal to the number of dimension of the problem" ) {
      CHECK_THROWS_AS(hookeJeevesAlgorithm.setInitialStepSize(arma::randu<arma::Mat<double>>(std::uniform_int_distribution<arma::uword>(3, 10)(mant::Rng::getGenerator())) * 200 - 100), std::logic_error);
      CHECK_THROWS_AS(hookeJeevesAlgorithm.setInitialStepSize(arma::randu<arma::Mat<double>>(1) * 200 - 100), std::logic_error);
    }

    SECTION(
          "Throws an exception, if the StepSizeDecrease lower than zero" ) {
      CHECK_THROWS_AS(hookeJeevesAlgorithm.setStepSizeDecrease({-0.001, -8.5}), std::logic_error);
    }

    SECTION(
          "Throws an exception, if the StepSizeDecrease higher than one" ) {
      CHECK_THROWS_AS(hookeJeevesAlgorithm.setStepSizeDecrease({1.8, 8.5}), std::logic_error);
    }

    SECTION(
          "Throws an exception, if the StepSizeDecrease zero" ) {
      CHECK_THROWS_AS(hookeJeevesAlgorithm.setStepSizeDecrease({0.0, 0.0}), std::logic_error);
    }

    SECTION(
          "Throws not an exception, if the StepSizeDecrease one" ) {
      CHECK_NOTHROW(hookeJeevesAlgorithm.setStepSizeDecrease({1.0, 1.0}));

    }
  }

  SECTION(
      ".toString") {
    SECTION(
        "Returns a (filesystem friendly) name for the class.") {
      CHECK(mant::HookeJeevesAlgorithm(optimisationProblem).toString() ==
            "hooke_jeeves_algorithm");
    }
  }
}
