// Catch
#include <catch.hpp>
#include <catchExtension.hpp>

// C++ Standard Library
#include <memory>
#include <random>

// Mantella
#include <mantella>

class TestHillClimbing : public mant::HillClimbing {
 public:
  TestHillClimbing(
      const std::shared_ptr<mant::OptimisationProblem> optimisationProblem)
      : mant::HillClimbing(optimisationProblem),
        neighboursIndex_(0) {
  }

  void setVelocitys(
      const arma::Mat<double>& neighbours) {
    neighbours_ = neighbours;
  }

 protected:
  arma::Col<double> getRandomNeighbour(
      const arma::Col<double>& parameter,
      const arma::Col<double>& minimalDistance,
      const arma::Col<double>& maximalDistance) override {
    return neighbours_.col(neighboursIndex_++);
  }

  arma::uword neighboursIndex_;
  arma::Mat<double> neighbours_;
};

TEST_CASE(
    "HillClimbing") {
  SECTION(
      ".setMaximalStepSize") {
    std::shared_ptr<mant::OptimisationProblem> optimisationProblem(new mant::bbob::SphereFunction(2));
    mant::HillClimbing hillClimbing(optimisationProblem);
    hillClimbing.setMaximalNumberOfIterations(1000);

<<<<<<< HEAD
    SECTION("Test default value"){
      mant::storeSamplingProgress = true;
      hillClimbing.optimise();
      mant::storeSamplingProgress = false;

      std::vector<std::pair<arma::Col<double>, double>> actualSamples = hillClimbing.getSamplingProgress();
      arma::Col<double> expectedMaximalStepSize = ((optimisationProblem->getUpperBounds() - optimisationProblem->getLowerBounds()) / 10.0);
       std::pair<arma::Col<double>, double> bestParameter = actualSamples.at(0);
      arma::Col<double>::fixed<999> lengths;
      for (arma::uword i = 1; i < actualSamples.size(); i++) {
        arma::Col<double> vector = actualSamples.at(i).first - bestParameter.first;
        lengths.at(i-1) = arma::norm(vector);
        if (bestParameter.second > actualSamples.at(i).second) {
          bestParameter = actualSamples.at(i);
        }
      }

      CHECK(lengths.max() <= arma::norm(expectedMaximalStepSize));
    }

    SECTION("Test with parameter") {
      arma::Col<double> maximalStepSize({0.5, 0.000001});
      hillClimbing.setMaximalStepSize(maximalStepSize);
      mant::storeSamplingProgress = true;
      hillClimbing.optimise();
      mant::storeSamplingProgress = false;

      std::vector<std::pair<arma::Col<double>, double>> actualSamples = hillClimbing.getSamplingProgress();
      std::pair<arma::Col<double>, double> bestParameter = actualSamples.at(0);
      arma::Col<double>::fixed<999> lengths;
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

  SECTION(".optimise") {
    std::shared_ptr<mant::OptimisationProblem> optimisationProblem(new mant::bbob::SphereFunction(2));
    mant::HillClimbing hillClimbing(optimisationProblem);
    hillClimbing.setMaximalNumberOfIterations(1000);

    SECTION("Test uniform distribution"){

      SECTION("Maximal step size are equal per dimmension"){
        mant::storeSamplingProgress = true;
        hillClimbing.optimise();
        mant::storeSamplingProgress = false;

        std::vector<std::pair<arma::Col<double>, double>> actualSamples = hillClimbing.getSamplingProgress();
        arma::Col<double> expectedMaximalStepSize = ((optimisationProblem->getUpperBounds() - optimisationProblem->getLowerBounds()) / 10.0);
        std::pair<arma::Col<double>, double> bestParameter = actualSamples.at(0);
        arma::Col<double>::fixed<999> angles;
        arma::Col<double>::fixed<999> lengths;
        for (arma::uword i = 1; i < actualSamples.size(); i++) {
          arma::Col<double> vector = actualSamples.at(i).first - bestParameter.first;
          angles.at(i-1) = std::atan2(vector.at(0), vector.at(1));
          lengths.at(i-1) = arma::norm(vector);
          if (bestParameter.second > actualSamples.at(i).second) {
            bestParameter = actualSamples.at(i);
          }
        }

        arma::Col<arma::uword> histogram = arma::hist(angles, arma::linspace<arma::Col<double>>(-arma::datum::pi, arma::datum::pi, 20));
        CHECK(0.25 > static_cast<double>(histogram.max() - histogram.min()) / static_cast<double>(angles.n_elem));

        histogram = arma::hist(lengths, arma::linspace<arma::Col<double>>(0, arma::norm(expectedMaximalStepSize), 100));
        CHECK(0.25 > static_cast<double>(histogram.max() - histogram.min()) / static_cast<double>(lengths.n_elem));
      }
    }
=======
    SECTION(
        "Test default value") {
      //TODO
    }

    SECTION(
        "Test with parameter") {
      //TODO
    }
  }

  SECTION(
      ".optimise") {
    // TODO
>>>>>>> master
  }

  SECTION(
      "Exception tests") {
    std::shared_ptr<mant::OptimisationProblem> optimisationProblem(new mant::bbob::SphereFunction(2));
    mant::HillClimbing hillClimbing(optimisationProblem);

    SECTION(
        "Throws an exception, if the MaximalStepSize zero") {
      CHECK_THROWS_AS(hillClimbing.setMaximalStepSize({0, 0}), std::logic_error);
    }

    SECTION(
        "Throws an exception, if the size of MaximalStepSize is not equal to the number of dimension of the problem") {
      CHECK_THROWS_AS(hillClimbing.setMaximalStepSize(arma::randu<arma::Mat<double>>(std::uniform_int_distribution<arma::uword>(3, 10)(mant::Rng::getGenerator())) * 200 - 100), std::logic_error);
      CHECK_THROWS_AS(hillClimbing.setMaximalStepSize(arma::randu<arma::Mat<double>>(1) * 200 - 100), std::logic_error);
    }
  }

  SECTION(
      ".toString") {
    SECTION(
        "Returns the expected class name.") {
      std::shared_ptr<mant::OptimisationProblem> optimisationProblem(new mant::bbob::SphereFunction(2));
      CHECK(mant::HillClimbing(optimisationProblem).toString() ==
            "hill_climbing");
    }
  }
}
