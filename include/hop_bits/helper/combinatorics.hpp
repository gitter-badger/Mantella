#pragma once

// C++ Standard Library
#include <vector>

// Armadillo
#include <armadillo>

namespace hop {
  std::vector<arma::Col<unsigned int>> getCombinationsWithoutRepetition(
      const unsigned int& numberOfElements,
      const unsigned int& combinationSize);
}
