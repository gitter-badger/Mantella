#pragma once

// Armadillo
#include <armadillo>

namespace mant {
  arma::Col<double> getOrdinaryLeastSquaresEstimate(
      const arma::Mat<double>& parameters,
      const arma::Row<double>& objectiveValues);

  arma::Col<double> getGeneralisedLeastSquaresEstimate(
      const arma::Mat<double>& parameters,
      const arma::Row<double>& objectiveValues,
      const arma::Mat<double>& variance);
}
