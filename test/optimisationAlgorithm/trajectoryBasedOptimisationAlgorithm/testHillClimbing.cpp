// Catch
#include <catch.hpp>
#include <catchExtension.hpp>

// C++ Standard Library
#include <memory>
#include <random>

// Mantella
#include <mantella>

extern std::string testDirectory;

class TestHillClimbing : public mant::HillClimbing {
 public:
  TestHillClimbing(
      const std::shared_ptr<mant::OptimisationProblem> optimisationProblem)
      : mant::HillClimbing(optimisationProblem),
        neighboursIndex_(0) {
    }

    void setVelocities(
        const arma::Mat<double>& neighbours) {
      neighbours_ = neighbours;
    }

  protected:
    arma::Col<double> getRandomNeighbour(
        const arma::Col<double>& parameter,
        const arma::Col<double>& minimalDistance,
        const arma::Col<double>& maximalDistance) override {
      return (parameter + neighbours_.col(neighboursIndex_++));
    }
    
  arma::uword neighboursIndex_;
  arma::Mat<double> neighbours_;
};


TEST_CASE("HillClimbing", "" ) {
  std::shared_ptr<mant::OptimisationProblem> optimisationProblem(new mant::bbob::SphereFunction(2));
  mant::HillClimbing hillClimbing(optimisationProblem);
  hillClimbing.setMaximalNumberOfIterations(10000);

  SECTION(".setMaximalStepSize") {

    SECTION("Test default value" ) {
      mant::recordSamples = true;
      hillClimbing.optimise();
      mant::recordSamples = false;

      std::vector<std::pair<arma::Col<double>, double>> actualSamples = hillClimbing.getSamplingHistory();
      arma::Col<double> expectedMaximalStepSize = ((optimisationProblem->getUpperBounds() - optimisationProblem->getLowerBounds()) / 10.0);
      std::pair<arma::Col<double>, double> bestParameter = actualSamples.at(0);
      arma::Col<double> lengths(hillClimbing.getMaximalNumberOfIterations() - 1);

      for (arma::uword i = 1; i < actualSamples.size(); i++) {
        arma::Col<double> vector = actualSamples.at(i).first - bestParameter.first;
        lengths.at(i-1) = arma::norm(vector);
        if (bestParameter.second > actualSamples.at(i).second) {
          bestParameter = actualSamples.at(i);
        }
      }
      CHECK(lengths.max() <= arma::norm(expectedMaximalStepSize));
    }

    SECTION("Test with parameter" ) {
      arma::Col<double> maximalStepSize({0.5, 0.000001});
      hillClimbing.setMaximalStepSize(maximalStepSize);
      mant::recordSamples = true;
      hillClimbing.optimise();
      mant::recordSamples = false;

      std::vector<std::pair<arma::Col<double>, double>> actualSamples = hillClimbing.getSamplingHistory();
      std::pair<arma::Col<double>, double> bestParameter = actualSamples.at(0);
      arma::Col<double> lengths(hillClimbing.getMaximalNumberOfIterations() - 1);
      for (arma::uword i = 1; i < actualSamples.size(); i++) {
        arma::Col<double> vector = actualSamples.at(i).first - bestParameter.first;
        lengths.at(i-1) = arma::norm(vector);
        if (bestParameter.second > actualSamples.at(i).second) {
          bestParameter = actualSamples.at(i);
        }
      }
      CHECK(lengths.max() <= arma::norm(maximalStepSize));
    }
  }

  SECTION(".setMinimalStepSize") {

    SECTION("Test default value" ) {
      mant::recordSamples = true;
      hillClimbing.optimise();
      mant::recordSamples = false;

      std::vector<std::pair<arma::Col<double>, double>> actualSamples = hillClimbing.getSamplingHistory();
      std::pair<arma::Col<double>, double> bestParameter = actualSamples.at(0);
      arma::Col<double> lengths(hillClimbing.getMaximalNumberOfIterations() - 1);

      for (arma::uword i = 1; i < actualSamples.size(); i++) {
        arma::Col<double> vector = actualSamples.at(i).first - bestParameter.first;
        lengths.at(i-1) = arma::norm(vector);
        if (bestParameter.second > actualSamples.at(i).second) {
          bestParameter = actualSamples.at(i);
        }
      }
      CHECK(lengths.min() >= 0 );
    }

    SECTION("Test with parameter" ) {
      arma::Col<double> minimalStepSize({0.5, 0.000001});
      hillClimbing.setMinimalStepSize(minimalStepSize);
      mant::recordSamples = true;
      hillClimbing.optimise();
      mant::recordSamples = false;

      std::vector<std::pair<arma::Col<double>, double>> actualSamples = hillClimbing.getSamplingHistory();
      std::pair<arma::Col<double>, double> bestParameter = actualSamples.at(0);
      arma::Col<double> lengths(hillClimbing.getMaximalNumberOfIterations() - 1);
      for (arma::uword i = 1; i < actualSamples.size(); i++) {
        arma::Col<double> vector = actualSamples.at(i).first - bestParameter.first;
        lengths.at(i-1) = arma::norm(vector);
        if (bestParameter.second > actualSamples.at(i).second) {
          bestParameter = actualSamples.at(i);
        }
      }
      CHECK(lengths.max() >= arma::norm(minimalStepSize));
    }
  }

  SECTION(".optimise") {

    SECTION("Test uniform distribution"){

      SECTION("Maximal and Minimal step size are equal per dimmension" ) {
        mant::recordSamples = true;
        hillClimbing.optimise();
        mant::recordSamples = false;

        std::vector<std::pair<arma::Col<double>, double>> actualSamples = hillClimbing.getSamplingHistory();
        arma::Col<double> expectedMaximalStepSize = ((optimisationProblem->getUpperBounds() - optimisationProblem->getLowerBounds()) / 10.0);

        std::pair<arma::Col<double>, double> bestParameter = actualSamples.at(0);
        arma::Col<double> angles(hillClimbing.getMaximalNumberOfIterations() - 1);
        arma::Col<double> lengths(hillClimbing.getMaximalNumberOfIterations() - 1);

        for (arma::uword i = 1; i < actualSamples.size(); i++) {
          arma::Col<double> vector = actualSamples.at(i).first - bestParameter.first;
          angles.at(i-1) = std::atan2(vector.at(0), vector.at(1));
          lengths.at(i-1) = arma::norm(vector);
          if (bestParameter.second > actualSamples.at(i).second) {
            bestParameter = actualSamples.at(i);
          }
        }

        IS_UNIFORM(angles, -arma::datum::pi, arma::datum::pi);
        IS_UNIFORM(lengths, 0.0, arma::norm(expectedMaximalStepSize));

      }

      SECTION("Maximal step size are not equal per dimmension" ) {
        arma::Col<double> expectedMaximalStepSize ({1.85, 0.589});
        hillClimbing.setMaximalStepSize(expectedMaximalStepSize);
        mant::recordSamples = true;
        hillClimbing.optimise();
        mant::recordSamples = false;

        std::vector<std::pair<arma::Col<double>, double>> actualSamples = hillClimbing.getSamplingHistory();
        std::pair<arma::Col<double>, double> bestParameter = actualSamples.at(0);
        arma::Col<double> angles(hillClimbing.getMaximalNumberOfIterations() - 1);
        arma::Col<double> lengths(hillClimbing.getMaximalNumberOfIterations() - 1);
        for (arma::uword i = 1; i < actualSamples.size(); i++) {
          arma::Col<double> vector = actualSamples.at(i).first - bestParameter.first;
          angles.at(i-1) = std::atan2(vector.at(0), vector.at(1));
          lengths.at(i-1) = arma::norm(vector);
          if (bestParameter.second > actualSamples.at(i).second) {
            bestParameter = actualSamples.at(i);
          }
        }

        IS_UNIFORM(angles, -arma::datum::pi, arma::datum::pi);
        IS_UNIFORM(lengths, 0.0, arma::norm(expectedMaximalStepSize));
      }

    }

    SECTION("Test the algorithm´s workflow"){
      arma::Col<double> parameterTranslation;
      REQUIRE(parameterTranslation.load(testDirectory + "/data/optimisationAlgorithm/trajectoryBasedAlgrorithm/hillClimbing/hillclimbing_optimise_1x2.parameterTranslation"));

      arma::Col<double> objectiveValueTranslation;
      REQUIRE(objectiveValueTranslation.load(testDirectory + "/data/optimisationAlgorithm/trajectoryBasedAlgrorithm/hillClimbing/hillclimbing_optimise_1x1.objectiveValueTranslation"));

      arma::Mat<double> velocities;
      REQUIRE(velocities.load(testDirectory + "/data/optimisationAlgorithm/trajectoryBasedAlgrorithm/hillClimbing/hillclimbing_optimise_11x2.velocities"));

      arma::Col<double> initParameter;
      REQUIRE(initParameter.load(testDirectory + "/data/optimisationAlgorithm/trajectoryBasedAlgrorithm/hillClimbing/hillclimbing_optimise_1x2.initParameter"));

      arma::Col<double> bestValue;
      REQUIRE(bestValue.load(testDirectory + "/data/optimisationAlgorithm/trajectoryBasedAlgrorithm/hillClimbing/hillclimbing_optimise_1x1.bestValue"));

      arma::Mat<double> parameter;
      REQUIRE(parameter.load(testDirectory + "/data/optimisationAlgorithm/trajectoryBasedAlgrorithm/hillClimbing/hillclimbing_optimise_12x2.expected"));

      optimisationProblem->setObjectiveValueTranslation(objectiveValueTranslation(0));
      optimisationProblem->setParameterTranslation(parameterTranslation);
      optimisationProblem->setAcceptableObjectiveValue(bestValue(0)-1);

      TestHillClimbing testHillClimbing(optimisationProblem);
      testHillClimbing.setInitialParameter(initParameter);
      testHillClimbing.setMaximalNumberOfIterations(parameter.n_cols);
      testHillClimbing.setVelocities(velocities);

      mant::recordSamples = true;
      testHillClimbing.optimise();
      mant::recordSamples = false;

      CHECK(testHillClimbing.getBestObjectiveValue() == Approx(bestValue(0)));

      std::vector<arma::Col<double>> expected;
      for (arma::uword i=0; i < parameter.n_cols; i++) {
        expected.push_back(parameter.col(i));
      }
      HAS_SAME_PARAMETERS(testHillClimbing.getSamplingHistory(),expected);

    }
  }

  SECTION("Exception tests") {

    SECTION(
          "Throws an exception, if the MaximalStepSize zero" ) {
      CHECK_THROWS_AS(hillClimbing.setMaximalStepSize({0, 0}), std::logic_error);
    }

    SECTION(
          "Throws an exception, if the MaximalStepSize is lower than MinimalStepSize" ) {
      hillClimbing.setMinimalStepSize({4.0, 4.2});
      hillClimbing.setMaximalStepSize({1.0, 4.0});
      CHECK_THROWS_AS(hillClimbing.optimise(), std::logic_error);
    }

    SECTION(
          "Throws an exception, if the size of MaximalStepSize is not equal to the number of dimension of the problem" ) {
      CHECK_THROWS_AS(hillClimbing.setMaximalStepSize(arma::randu<arma::Mat<double>>(std::uniform_int_distribution<arma::uword>(3, 10)(mant::Rng::getGenerator())) * 200 - 100), std::logic_error);
      CHECK_THROWS_AS(hillClimbing.setMaximalStepSize(arma::randu<arma::Mat<double>>(1) * 200 - 100), std::logic_error);
    }

    SECTION(
          "Throws an exception, if the MinimalStepSize lower than zero" ) {
      CHECK_THROWS_AS(hillClimbing.setMinimalStepSize({-0.001, -8.5}), std::logic_error);
    }

    SECTION(
          "Throws an exception, if the MinimalStepSize is higher than MaximalStepSize" ) {
      hillClimbing.setMaximalStepSize({4.0, 4.2});
      hillClimbing.setMinimalStepSize({6.0, 6.0});
      CHECK_THROWS_AS(hillClimbing.optimise(), std::logic_error);
    }

    SECTION(
          "Throws an exception, if the size of MinimalStepSize is not equal to the number of dimension of the problem" ) {
      CHECK_THROWS_AS(hillClimbing.setMinimalStepSize(arma::randu<arma::Mat<double>>(std::uniform_int_distribution<arma::uword>(3, 10)(mant::Rng::getGenerator())) * 200 - 100), std::logic_error);
      CHECK_THROWS_AS(hillClimbing.setMinimalStepSize(arma::randu<arma::Mat<double>>(1) * 200 - 100), std::logic_error);
    }

  }

  SECTION(".toString") {
    SECTION(
        "Returns a (filesystem friendly) name for the class.") {
      CHECK(mant::HillClimbing(optimisationProblem).toString() ==
            "hill_climbing" );
    }
  }
}

